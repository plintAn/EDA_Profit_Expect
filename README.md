# EDA_Profit_Expect

# 주택 가격 예측 - 고급 회귀 기법
* 판매 가격을 예측하고 기능 엔지니어링, RF 및 그래디언트 부스팅 연습

데이터셋 출처(https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

# 1.데이터 로드
```python
import pandas as pd
df = pd.read_csv('./train.csv')
df.head()
```

# 2.데이터 확인 및 전처리

```python

Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	RL	65.0	8450	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500
1	2	20	RL	80.0	9600	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500
2	3	60	RL	68.0	11250	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500
3	4	70	RL	60.0	9550	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000
4	5	60	RL	84.0	14260	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000
5 rows × 81 columns
```

- 1460 rows × 81 columns
- 열이 전부 영어로 돼 있습니다.
- Null, Nan 값이 다수로 보입니다.
- `(건물 연결 거리)LotFrontage' 값은 수치형 데이터지만 결측값이 종종 보인다.

### 모든 열을 한국어로 재구성

```python
df.columns = ['Id', '건물 클래스', '일반구역 분류', '건물 연결 거리', '부지 크기', '도로 접근 유형', '골목 진입 방식', 
              '속성 모양', '평탄도', '유틸리티 유형', '로트 구성', '속성 기울기', '인근 지역', '주요 도로/철도 인접1', 
              '주요 도로/철도 인접2', '주거형태', '주거 스타일', '재료 및 마감 품질', '전반적인 상태', '건설 연도', '리모델링 연도', ....
              ....
              '부동산 판매 가격']

```

## 2.1 결측값을 제거(`(건물 연결 거리)LotFrontage'제외)

* `(건물 연결 거리)LotFrontage' 변수는 유의미한 상관관계를 보여줄 것 같아 제외하고 결측값 제거한다

```python
# 수치형 열의 결측값을 0으로 대체
for column in df.select_dtypes(include=[np.number]).columns:
    df[column].fillna(0, inplace=True)

# 문자열 열에서 결측값이 있는 경우 해당 열 삭제
for column in df.select_dtypes(include=[object]).columns:
    if df[column].isnull().any():
        df.drop(column, axis=1, inplace=True)
```

결측값 제거 후 데이터 확인작업

```python
df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 65 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Id            1460 non-null   int64  
 1   건물 클래스        1460 non-null   int64  
 2   일반구역 분류       1460 non-null   object 
 3   건물 연결 거리      1460 non-null   float64
 4   부지 크기         1460 non-null   int64  
 5   도로 접근 유형      1460 non-null   object 
 6   속성 모양         1460 non-null   object 
...
 63  판매 조건         1460 non-null   object 
 64  부동산 판매 가격     1460 non-null   int64  
dtypes: float64(3), int64(35), object(27)
memory usage: 741.5+ KB
```

* 81 rows ==> 65 rows로 상당한 변화를 보여준다.


### 65개 변수 정리

변수는 다음과 같습니다. 65개의 변수는 범위형, 수치형 등 다양한 형태로 존재하는데, 이 값 중 '부동산 판매 가격'에 가장 큰 영향을 미치는 변수를 선택해야 합니다.



| 변수            | 설명                           | 변수           | 설명                          |
|-----------------|-------------------------------|----------------|------------------------------|
| SalePrice       | 부동산 판매 가격                | MSSubClass     | 건물 클래스                   |
| MSZoning        | 구역 분류                     | LotFrontage    | 연결된 거리 길이              |
| LotArea         | 부지 크기                      | Street         | 도로 접근 유형                |
.....
| MoSold          | 판매 월                        | YrSold         | 판매 연도                      |
| SaleType        | 판매 유형                      | SaleCondition  | 판매 상태                      |


가격에 가장 큰 영향을 미치는 변수를 선택하기 위해서는 다양한 통계적 방법 또는 기계 학습 기법을 활용할 수 있는데, 주요 방법은 다음과 같다.

* 상관 계수: 수치형 변수들 간의 선형 관계를 평가
* 트리 기반 모델의 기능 중요도: 결정 트리, 랜덤 포레스트, XGBoost 등의 트리 기반 알고리즘은 변수의 중요도를 제공
* 일변량 특징 선택: 단변량 통계 검정을 통해 각 변수가 출력 변수(가격)과 얼마나 잘 관련되어 있는지 평가

### 트리 기반 모델의 특성 중요도 방법 선택

* 트리 기반 알고리즘은 수치형 뿐만 아니라 범주형 변수에도 적용할 수 있다.
* 변수 간의 상호 작용을 포착할 수 있다.
* 중요도 순으로 변수를 랭킹할 수 있어서 선택의 폭이 넓다.



```python
from sklearn.ensemble import RandomForestRegressor

# 중요도 출력 형식 설정
pd.options.display.float_format = '{:.7f}'.format

# 데이터 준비 (범주형 변수를 원-핫 인코딩 or 라벨 인코딩으로 처리해야 함)
# 예시로는 원-핫 인코딩을 사용합니다.
df_encoded = pd.get_dummies(df)

# X: 설명 변수, y: 목표 변수
X = df_encoded.drop("부동산 판매 가격", axis=1)
y = df_encoded["부동산 판매 가격"]

# 랜덤 포레스트 모델 학습
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# 변수 중요도 순으로 정렬
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)
```

```python
             Feature  Importance
4         재료 및 마감 품질   0.5841559
16        지상 거실 평방피트   0.1089196
12         총 지하 평방피트   0.0409507
14           2층 평방피트   0.0341491
9        지하 마감 평방피트1   0.0321642
..               ...         ...
135   외부 피복재1_CBlock   0.0000000
169        외장재 상태_Po   0.0000000
53    유틸리티 유형_NoSeWa   0.0000000
150   외부 피복재2_CBlock   0.0000000
138  외부 피복재1_ImStucc   0.0000000

[219 rows x 2 columns]
```
### 결과가 219 rows x 2 columns 으로 65 rows에서 거의 3개 가량 늘어난 이유

코드에 설명한대로 One-Hot Encoding 원-핫 인코딩을 사용했기 때문입니다. 예시로 one-hot encoding 후 '도로 접근 유형_도로1', '도로 접근 유형_도로2', '도로 접근 유형_도로3' 세 개의 새로운 열이 늘어나게 됩니다.


'부동산 판매 가격'에 가장 큰 영향을 미치는 변수는 상위 10개

```python
Feature	Importance
4	재료 및 마감 품질	0.5841559
16	지상 거실 평방피트	0.1089196
12	총 지하 평방피트	0.0409507
14	2층 평방피트	0.0341491
9	지하 마감 평방피트1	0.0321642
13	1층 평방피트	0.0248908
26	차고 크기	0.0233101
27	차고 면적	0.0132368
3	부지 크기	0.0131706
6	건설 연도	0.0079752
```


통상 변수는누적 중요도(누적 90% ~ 95%)를 선택하는 경우가 많다. 하지만 이번 프로젝트는 DEA이므로 조금 더 탐구적인 방법인 상관 분석을 통해 변수 선택을 진행해보겠습니다.

상위 5개 변수와 '부동산 판매 가격' 의 상관 분석

### 피어슨 상관 계수

```python
top_5_features = ['재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트', '2층 평방피트', '지하 마감 평방피트1']

correlations = df[top_5_features + ['부동산 판매 가격']].corr()['부동산 판매 가격'].drop('부동산 판매 가격')

print(correlations)
```

```python
재료 및 마감 품질    0.7909816
지상 거실 평방피트    0.7086245
총 지하 평방피트     0.6135806
2층 평방피트       0.3193338
지하 마감 평방피트1   0.3864198
Name: 부동산 판매 가격, dtype: float64
```

### 변수 3개 선택

OutPut은 다음과 같이 확인해 볼 수 있었는데, '재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트' 세 변수가 강한 양의 상관 계수를 보이므로 3개를 선택

'재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트'

## 2-2 이상치 제거

### 사분위수 범위(IQR) 이상치 제거 후 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt

# IQR 방식을 사용한 이상치 제거 함수
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치가 아닌 데이터만 반환
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# '부동산 판매 가격'의 이상치 제거
df_clean = remove_outliers(df, '부동산 판매 가격')

```
#### 시각화
```python

# 시각화
# '재료 및 마감 품질' vs '부동산 판매 가격'
plt.figure(figsize=(10, 6))
sns.boxplot(x='재료 및 마감 품질', y='부동산 판매 가격', data=df_clean)
plt.title('재료 및 마감 품질에 따른 부동산 판매 가격')
plt.show()

# '지상 거실 평방피트' vs '부동산 판매 가격'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='지상 거실 평방피트', y='부동산 판매 가격', data=df_clean)
plt.title('지상 거실 평방피트에 따른 부동산 판매 가격')
plt.show()

# '총 지하 평방피트' vs '부동산 판매 가격'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='총 지하 평방피트', y='부동산 판매 가격', data=df_clean)
plt.title('총 지하 평방피트에 따른 부동산 판매 가격')
plt.show()

```
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/7ba10c39-8094-4591-83fb-785f48f35588)
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/40f224e8-1f2d-438c-8bf9-f7934be4b3d6)
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/dfde39db-f1d4-4df9-b619-10ceb5867491)

## 2-3 변수 변환

### 로그 변환(logit transformation)

로그 변환은 치우친 분포의 데이터를 보다 정규 분포에 가깝게 만들기 위해 사용

```python
import numpy as np

# '부동산 판매 가격'에 로그 변환 적용
df['log_부동산 판매 가격'] = np.log1p(df['부동산 판매 가격'])
```

### 표준화(Standardization)

표준화는 각 변수의 평균을 0, 표준 편차를 1로 변환하는 과정

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['지상 거실 평방피트', '총 지하 평방피트']] = scaler.fit_transform(df[['지상 거실 평방피트', '총 지하 평방피트']])

```

### 정규화(Normalization)

정규화는 각 변수의 범위를 [0, 1]로 조정

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
df['건설 연도'] = min_max_scaler.fit_transform(df[['건설 연도']])

```

### 이진수 변환

앞서 진행한 one-hot encording 으로 인해 '재료 및 마감 품질'로 인해 카테고리별로 나누어 True, False로 이미 변환 돼 있다.

```python
재료 및 마감 품질_1: [False  True]
재료 및 마감 품질_2: [False  True]
재료 및 마감 품질_3: [False  True]
재료 및 마감 품질_4: [False  True]
재료 및 마감 품질_5: [False  True]
재료 및 마감 품질_6: [False  True]
재료 및 마감 품질_7: [ True False]
재료 및 마감 품질_8: [False  True]
재료 및 마감 품질_9: [False  True]
재료 및 마감 품질_10: [False  True]
```


# 3.탐색적 데이터 분석(EDA)

탐색적 데이터 분석 (EDA)는 데이터를 이해하고, 데이터의 패턴을 탐색하며, 예상되는 문제점이나 이상치를 발견하는 데 가장 중요한 단계이다. 여기서는 상관 분석과 시각화에 초점을 두어 분석합니다.
* 상관 분석: 각 특성과 목표 변수(부동산 판매 가격) 간의 상관관계를 확인합니다. 높은 상관계수를 갖는 특성은 중요한 요인으로 간주될 수 있습니다.
* 시각화: 각 특성의 분포, 목표 변수와의 관계 등을 시각화하여 데이터에 대한 통찰력을 얻습니다.
* 변환된 '재료 및 마감 품질'의 분포를 확인합니다.
* '지상 거실 평방피트'와 '총 지하 평방피트'의 분포를 확인합니다.
* 이 변수들과 목표 변수(예: '부동산 판매 가격') 간의 관계를 시각화합니다.
  
## 3.1 기술 통계(Descriptive Statistics):

* 변환된 '재료 및 마감 품질'와 다른 두 변수에 대한 통계적 요약을 제공

```python
print(df[['재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트']].describe())
```

## 3.1 시각화


```python
print(df[['재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트']].describe())
```

```python
        재료 및 마감 품질   지상 거실 평방피트    총 지하 평방피트
count 1460.0000000 1460.0000000 1460.0000000
mean     5.9760274   -0.0000000    0.0000000
std      1.4726571    1.0003426    1.0003426
min      0.0000000   -2.2491201   -2.4111669
25%      5.0000000   -0.7347485   -0.5966855
50%      6.0000000   -0.0979700   -0.1503334
75%      7.0000000    0.4974036    0.5491227
max      9.0000000    7.8555744   11.5209492
```

### 3.1.1 '재료 및 마감 품질' 시각화

* 원-핫 인코딩 전처리 작업으로 인해 재료 및 마감 품질_1 ~ 재료 및 마감 품질_10 으로 나누어져 있기에 다시 범주형으로 묶습니다

```python
def get_category(row):
    for col, value in row.items():
        if '재료 및 마감 품질_' in col and value == 1:
            return int(col.split('_')[-1])
    return None

df['재료 및 마감 품질'] = df.apply(get_category, axis=1)
```
시각화 진행

```python
sns.countplot(data=df, x='재료 및 마감 품질')
plt.title('재료 및 마감 품질 분포')
plt.show()
```

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/0eea321e-d077-4df7-ab45-c78bbf85dafa)

### 3.1.2 '지상 거실 평방피트'의  시각화

```python
# 2. '지상 거실 평방피트'의 분포 확인
living_space = np.array(df['지상 거실 평방피트'].dropna())  # NaN값 제거 후 배열로 변환
plt.hist(living_space, bins=50, density=True, alpha=0.6, color='b')
plt.title("'지상 거실 평방피트' 분포")
plt.xlabel("'지상 거실 평방피트'")
plt.ylabel("확률 밀도")
plt.show()
```

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/13530e1e-e075-4c47-85a5-9fd88b1ff352)

폰트 꺠짐 주의
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
```

### 3.1.3 '총 지하 평방피트'의  시각화

```python
# 3. '총 지하 평방피트'의 분포 확인
basement_space = np.array(df['총 지하 평방피트'].dropna())  # NaN값 제거 후 배열로 변환
plt.hist(basement_space, bins=50, density=True, alpha=0.6, color='g')
plt.title("'총 지하 평방피트' 분포")
plt.xlabel("'총 지하 평방피트'")
plt.ylabel("확률 밀도")
plt.show()
```

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/852e3e45-329b-4c7f-a3dd-7cd762ea0334)



* 시각화: 각 특성의 분포, 목표 변수와의 관계 등을 시각화하여 데이터에 대한 통찰력을 얻습니다.
* 변환된 '재료 및 마감 품질'의 분포를 확인합니다.
* '지상 거실 평방피트'와 '총 지하 평방피트'의 분포를 확인합니다.
* 이 변수들과 목표 변수(예: '부동산 판매 가격') 간의 관계를 시각화합니다.


# 4.모델링

* 특성 선택: 특성 선택 기법(예: 재귀적 특성 제거)을 사용하여 가장 중요한 특성을 선택할 수 있습니다.
* 선형 회귀 분석: 각 특성의 가중치를 확인하여 그 중요성을 평가합니다. 높은 가중치를 갖는 특성은 목표 변수에 큰 영향을 미칠 수 있습니다.
* 기타 모델: 랜덤 포레스트, 그래디언트 부스팅 등의 알고리즘도 중요한 특성을 선택하는 데 도움이 될 수 있습니다. 이러한 모델에서 제공하는 특성 중요도를 확인하여 주요 요인을 식별합니다.

# 5.모델 검증

* 교차 검증: 모델의 안정성을 확인하기 위해 교차 검증을 수행합니다.
* 성능 지표: RMSE, R^2 등의 성능 지표를 사용하여 모델의 예측 성능을 평가합니다.

# 6.결과 해석 및 고찰

* 모델링을 통해 얻은 결과를 바탕으로 주요 요인에 대한 해석을 수행합니다.
* 어떤 특성이 부동산 가격에 큰 영향을 미치는지, 그 영향의 방향성(양의 영향 또는 음의 영향)과 크기를 평가합니다.

# 7.결론 도출 및 추천

* 분석 결과를 바탕으로 부동산 가격에 미치는 주요 요인과 그 영향력에 대한 결론을 도출합니다.
* 부동산 판매나 구매와 관련된 추천 사항을 제시할 수 있습니다.








