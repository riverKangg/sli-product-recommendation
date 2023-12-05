## 보험 상품 추천 모델 개발

| [과제기획](#1.과제기획) | [분석설계](#분석설계) | [모델링](#모델링) | [테스트](#테스트) |
|:---------------:|:-------------:|:-----------:|:-----------:|

---

## 과제 기획

#### 기존 상품 추천

- 기존 상품 추천은 상품 구매 여부만으로 연관분석으로 진행됨
    - 상품중분류 단위(ex. 암, 건강, 건강_간편 등)
    - 치아보험만 파일럿 진행
    - 고객카드에 제공되는 방식은 아직 구체적으로 나온게 없음

#### 신규 상품 추천 기획

- 다양한 고객 속성을 반영하여 상품 추천하고자 함
    - 우선 교류 데이터(화재, 증권, 카드)만 반영하고, 추후 마이데이터까지 확장 예정
- 상품 종류가 다양하고 보험 계약은 빈번하게 일어나는 이벤트가 아님. 계약 정보에는 카테고리형 데이터가 많음 → 매우 회소(sparse)한 데이터가 되어 일반적인 모델 학습이 어려움
-

---

## 분석 설계

### 대상자 선정

- 당사 고객 중 카드 혹은 증권 교류 데이터 동의 고객

### 방법론 설정

- 대부분의 추천 모델은 유저, 아이템, 구매 여부(평점)을 사용하여 추천함
    - Content Based Filtering(CBF), Collaborative Filtering(CF)
- 추천되는 보험 상품의 정보

### 타겟 설정

- 대상 : 현재 판매중인 상품 or 기존 로직 타겟 상품
- 상품코드 규칙
    - LP , LA,
    - 숫자앞네자리 : 상품
    - 숫자뒷세자리 : 개정 버전
    - LP0571005: 삼성간편종합보장보험(2202)(갱신형, 무배당) 건강하고 당당하게_1종 간편고지형
    - LP0571006: 삼성간편종합보장보험(2204)(갱신형, 무배당) 건강하고 당당하게_1종 간편고지형

### 데이터 구성

- 타겟 메타정보 : 타겟 상품을 나타낼 수 있는 정보
    - 상세담보코드, 보장진단코드 → 연금저축 상품의 특징을 나타내주지 못함.
- 고객 메타정보
- 기계약 메타정보

---

## 모델링

- 건강상품 압도적으로 많고 상품 카테고리별 특징을 넣어줄 수 있는 피처가 부족함. 건강/종신까지는 분류를 하지만 이외는 제대로 분류하지 못함. 
  → 건강(or종신까지)으로 분류된 대상자 제외하고 MultiClass-Classification 진행

---

## 결과 비교

- //
