# MPR (Mixed-Precision Recovery) 알고리즘 명세

구현 방식과 독립된 알고리즘 레벨 명세.  
Quest 포크 위에 다시 구현할 때 이 문서를 기준으로 삼는다.

---

## 1. 개요

GPU KV cache 용량이 시퀀스 전체를 담기에 부족할 때, evict된 블록을 CPU에 백업하고
decode step마다 digest 기반 score로 중요도를 추정하여 **다른 precision으로 선택적 recall**하는 방법.

```
고점수 블록 → FP16으로 recall   (attention 품질 최대 보존)
중점수 블록 → INT8으로 recall   (메모리/전송 절감)
저점수 블록 → INT4로 recall     (더 많은 절감)
최저점수    → skip              (해당 step 어텐션에서 제외)
```

---

## 2. 핵심 데이터

### 2.1 Digest (per-block, GPU 상주)

블록의 key tensor `[block_size, num_kv_heads, head_dim]`에서 추출한 요약.

```
digest_min: [num_kv_heads, head_dim]  — 각 채널의 최솟값
digest_max: [num_kv_heads, head_dim]  — 각 채널의 최댓값
```

**핵심 속성**: 블록이 CPU로 evict된 뒤에도 digest는 GPU에 남아 있어야 함.
score 추정에 실제 KV 데이터가 필요 없기 때문.

Digest 변형:
- `raw_minmax`: 직접 min/max
- `arkvale`: `center ± mean_abs_dist` (tighter bounds)
  - `center = (max + min) / 2`
  - `dist = mean(abs(keys - center), dim=token)`

### 2.2 CPU Backup (per-block)

블록 하나 `[2, block_size, num_kv_heads, head_dim]` (2 = K/V)를
evict 시점에 세 precision으로 **동시에** 인코딩해서 CPU에 보관.

#### FP16 payload
```
tensor: [2, block_size, num_kv_heads, head_dim]  dtype=float16/bfloat16
```
원본 복사본. 복원 시 exact.

#### INT8 payload
```
quantized: [2, block_size, num_kv_heads, head_dim]  dtype=int8
scale:     [2, block_size, num_kv_heads]             dtype=float32
```
인코딩:
```
scale[b, t, h] = max(abs(kv[b, t, h, :])) / 127
quantized = round(kv / scale).clamp(-127, 127)
```
복원:
```
kv_recovered = quantized.float() * scale  →  target_dtype
```
granularity: per-token-per-kv-head (scale 하나가 head_dim 채널 전체 공유)

#### INT4 payload
```
packed: [2, block_size, num_kv_heads, (head_dim+1)//2]  dtype=uint8
scale:  [2, block_size, num_kv_heads]                    dtype=float32
```
인코딩:
```
scale = max(abs(kv)) / 7
quantized = round(kv / scale).clamp(-7, 7)
# nibble packing: 짝수 인덱스 → 하위 4비트, 홀수 인덱스 → 상위 4비트
```
복원:
```
low  = (packed & 0x0F).to(int8) - 8    # sign-extend: [0..15] → [-8..7]
high = (packed >> 4).to(int8)  - 8     # 또는 ((packed >> 4) | 0xF0).view(int8)로 sign-extend
dequant = concat(low, high, interleave) * scale  →  target_dtype
```

---

## 3. Scoring (per-decode-step)

### 3.1 Score 공식 (Quest cuboid score)

query window `q: [num_q_heads, head_dim]`,
digest `d_min/d_max: [num_blocks, num_kv_heads, head_dim]`에 대해:

**per query head score:**
```python
# q를 kv head 그룹별로 reshape: [num_kv_heads, group_size, head_dim]
# 브로드캐스트 후:
score_per_qhead[block, q_head] = sum_over_dim(
    max(q[q_head] * d_max[block, q_head//group_size],
        q[q_head] * d_min[block, q_head//group_size])
)
# shape: [num_blocks, num_q_heads]
```

**GQA 집계 (block score로 collapse):**
```python
# step 1: query head group 내 max → per_kv_head_scores [num_blocks, num_kv_heads]
per_kv = per_qhead_scores.reshape(num_blocks, num_kv_heads, group_size).max(dim=2)

# step 2: kv head 간 max → block_scores [num_blocks]
block_scores = per_kv.max(dim=1)
```

근거: "같은 KV 그룹 내에서 어느 query head라도 높은 점수를 주면 FP16으로 recall"
= conservative union policy. monotonic threshold이므로 아래와 동치:
`tier(max_score) == max_tier(per_head_tiers)`

### 3.2 Query Window

단일 decode step의 query `q[0, :, 0, :]` 그대로 사용해도 되고,
rolling average (최근 N step의 query 평균)를 사용해도 됨.
rolling average가 score 안정성에 유리하지만 구현 복잡도 증가.

---

## 4. Precision 정책 (Threshold 기반)

```
score >= tier_high_threshold  →  FP16
score >= tier_mid_threshold   →  INT8
score >= tier_low_threshold   →  INT4
score <  tier_low_threshold   →  skip
```

제약:
```
tier_high >= tier_mid >= tier_low
```

threshold는 score 분포의 분위수로 설정하는 게 실용적
(예: p75 → FP16, p50 → INT8, p25 → INT4).

**Recall 예산 제약**: recall할 블록 수는 GPU capacity - (현재 쓰이는 슬롯 수)를 넘을 수 없음.
초과 시 precision 우선순위(FP16 > INT8 > INT4)로 잘라냄.

---

## 5. 통합 포인트 (Integration Hooks)

Quest 기반으로 구현할 때 어느 위치에서 무엇을 해야 하는지.

### 5.1 Prefill 이후 (시퀀스 초기화)

```
for each layer:
    완성된 블록마다:
        digest 계산 → GPU digest table에 저장
        GPU capacity 초과 시: evict → CPU에 3-tier backup 생성
```

### 5.2 Decode 매 step

```
for each layer:
    1. 모든 finalized 블록에 대해 score 계산 (digest만 있으면 됨)
    2. threshold policy로 tier 배정
    3. 예산 내 블록만 선택 (FP16 > INT8 > INT4 우선순위)
    4. 비거주 블록: GPU 슬롯 확보 (저점수 거주 블록 evict) → recall at assigned precision
    5. 새 토큰 K/V append
    6. attention: 거주 중인 블록 + 현재 블록만 대상
```

### 5.3 Eviction (슬롯 부족 시)

```
victim = argmin(last_score, GPU_resident_blocks)
CPU backup 생성 (fp16 + int8 + int4 동시)
digest는 GPU에 유지
GPU 슬롯 반환
```

---

## 6. 불변식 (Invariants)

1. 블록의 **식별자**(CPU backup key, digest lookup key)는 해당 블록이 GPU↔CPU 사이를 이동해도 변하지 않아야 함  
   → 식별자를 어떻게 구현하는지(sequence offset, hash, monotonic id 등)는 구현 결정 사항

2. digest는 블록이 GPU에서 evict된 후에도 GPU에 상주

3. recall 결과물은 항상 target_dtype (fp16/bf16)으로 dequantize됨  
   → mixed-dtype attention 없음, attention 연산 자체는 단일 dtype

4. 현재 쓰이고 있는 (미완성) 블록은 eviction 대상에서 제외

---

## 7. 구현 시 결정 사항 (Implementation-defined)

아래는 알고리즘 spec 범위 밖이고 구현에서 결정:

- Eviction 정책: 최저 score vs. LRU vs. 혼합
- Query window: 단일 step vs. rolling average, window 크기
- Digest 방식: raw_minmax vs. arkvale
- Backup 시점: eager (evict 즉시) vs. lazy (recall 요청 시)  
  → 현재 검증된 방식은 eager
- CPU 메모리 관리: pool pre-allocation vs. on-demand
- Recall 순서: finalized 블록 순 vs. score 순
