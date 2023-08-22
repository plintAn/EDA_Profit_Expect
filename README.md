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
4           재료 및 마감 품질   0.5836501
16          지상 거실 평방피트   0.1068863
12           총 지하 평방피트   0.0395690
14             2층 평방피트   0.0352247
9          지하 마감 평방피트1   0.0321172
..                 ...         ...
53      유틸리티 유형_NoSeWa   0.0000000
52      유틸리티 유형_AllPub   0.0000000
103  주요 도로/철도 인접2_RRNn   0.0000000
101  주요 도로/철도 인접2_RRAe   0.0000000
176         기초 유형_Wood   0.0000000

[219 rows x 2 columns]

[219 rows x 2 columns]
```
### 결과가 219 rows x 2 columns 으로 65 rows에서 거의 3개 가량 늘어난 이유

코드에 설명한대로 One-Hot Encoding 원-핫 인코딩을 사용했기 때문입니다. 예시로 one-hot encoding 후 '도로 접근 유형_도로1', '도로 접근 유형_도로2', '도로 접근 유형_도로3' 세 개의 새로운 열이 늘어나게 됩니다.


'부동산 판매 가격'에 가장 큰 영향을 미치는 변수는 상위 5개

```python
               Feature  Importance
4           재료 및 마감 품질   0.5836501
16          지상 거실 평방피트   0.1068863
12           총 지하 평방피트   0.0395690
14             2층 평방피트   0.0352247
9          지하 마감 평방피트1   0.0321172
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
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/aed7a75e-c5c0-4496-aff9-a865808ecc67)

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/42d5f451-41bb-45fd-b7b0-4ae721b3345f)

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/66ff891b-dd02-471f-b03f-b9a84fe91c63)

## 2-3 변수 변환

### 로그 변환(logit transformation)

로그 변환은 치우친 분포의 데이터를 보다 정규 분포에 가깝게 만들기 위해 사용

```python
import numpy as np

# '부동산 판매 가격'에 로그 변환 적용
df['log_부동산 판매 가격'] = np.log1p(df['부동산 판매 가격'])

df['log_부동산 판매 가격']
```

```python
0      12.2476991
1      12.1090164
2      12.3171712
3      11.8494048
4      12.4292202
          ...    
1455   12.0725470
1456   12.2548676
1457   12.4931333
1458   11.8644693
1459   11.9015902
Name: log_부동산 판매 가격, Length: 1460, dtype: float64
```


### 표준화(Standardization)

표준화는 각 변수의 평균을 0, 표준 편차를 1로 변환하는 과정

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['지상 거실 평방피트', '총 지하 평방피트']] = scaler.fit_transform(df[['지상 거실 평방피트', '총 지하 평방피트']])

df[['지상 거실 평방피트', '총 지하 평방피트']]
```
```python

지상 거실 평방피트	총 지하 평방피트
0	0.3703334	-0.4593025
1	-0.4825119	0.4664649
2	0.5150126	-0.3133688
3	0.3836591	-0.6873241
4	1.2993257	0.1996797
...	...	...
1455	0.2504021	-0.2381216
1456	1.0613666	1.1049252
1457	1.5696472	0.2156412
1458	-0.8327877	0.0469053
1459	-0.4939340	0.4527836
1460 rows × 2 columns
```


### 정규화(Normalization)

정규화는 각 변수의 범위를 [0, 1]로 조정

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
df['건설 연도'] = min_max_scaler.fit_transform(df[['건설 연도']])

df['건설 연도']

```
```python
0      0.9492754
1      0.7536232
2      0.9347826
3      0.3115942
4      0.9275362
          ...   
1455   0.9202899
1456   0.7681159
1457   0.5000000
1458   0.5652174
1459   0.6739130
Name: 건설 연도, Length: 1460, dtype: float64
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

```python
        재료 및 마감 품질   지상 거실 평방피트    총 지하 평방피트
count 1460.0000000 1460.0000000 1460.0000000
mean     6.0993151   -0.0000000    0.0000000
std      1.3829965    1.0003426    1.0003426
min      1.0000000   -2.2491201   -2.4111669
25%      5.0000000   -0.7347485   -0.5966855
50%      6.0000000   -0.0979700   -0.1503334
75%      7.0000000    0.4974036    0.5491227
max     10.0000000    7.8555744   11.5209492
```

## 3.1 시각화

### 3.1.1 '재료 및 마감 품질' 시각화

시각화 진행

```python
import seaborn as sns
import matplotlib.pyplot as plt


# Seaborn 스타일 및 컬러 팔레트 설정
sns.set_style("whitegrid")
sns.set_palette("viridis", n_colors=len(df['재료 및 마감 품질'].unique()))

plt.figure(figsize=(12, 7))

# 카운트 플롯 생성
# 폰트 깨짐 해결
plt.rcParams['font.family'] = 'Malgun Gothic'
ax = sns.countplot(data=df, x='재료 및 마감 품질', order=df['재료 및 마감 품질'].value_counts().index)

# 그래프 제목 및 레이블 설정
plt.title('재료 및 마감 품질 분포', fontsize=18)
plt.xlabel('재료 및 마감 품질', fontsize=16)
plt.ylabel('빈도수', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)

# 각 막대에 빈도수 텍스트 추가
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()

```

![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/24a148e2-414b-4f08-858d-06a09b9a5037)


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
* 
### 3.1.4 '부동산 판매 가격' 과 세 변수의분석

선택한 세 변수인 '재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트'와 '부동산 판매 가격' 간의 관계를 알아보겠습니다.

세 변수에 대한 scatter plot과 선형 회귀선을 추가

```python
# '부동산 판매 가격'과 세 변수 간의 관계 시각화
variables = ['재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트']

for var in variables:
    sns.scatterplot(x=df[var], y=df['부동산 판매 가격'], alpha=0.6)
    sns.regplot(x=df[var], y=df['부동산 판매 가격'], scatter=False, color='red')  # 회귀선 추가
    plt.title(f"{var}와 부동산 판매 가격의 관계")
    plt.show()
```
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/5c2ea41f-9c93-4ad7-aad7-cb60c73eba82)
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/7b72c224-2ab7-4521-b74b-28254add8bae)
![image](https://github.com/plintAn/EDA_Profit_Expect/assets/124107186/3f48d6cc-8f22-43ed-9e4a-02a175d742b3)





# 4.모델링

### 특성 선택:
 피어슨 상관 계수를 기반으로 이미 세 가지 주요 변수를 선택했습니다. 이 세 변수는 '부동산 판매 가격'과의 관계가 높습니다.
 
### 선형 회귀 분석: 
선형 회귀는 연속적인 목표 변수의 값을 예측하는 데 사용되며, 변수 간의 선형 관계를 가정합니다. 선택한 변수들은 부동산 판매 가격과 높은 상관관계를 가지고 있기 때문에 선형 회귀 분석에 적합합니다.

### 기타 모델: 
다른 모델도 고려될 수 있습니다. 예를 들어, 랜덤 포레스트, 그래디언트 부스팅, 신경망 등입니다. 이 모델들은 특성 간의 비선형 관계나 상호 작용을 포착하는 데 더 효과적일 수 있습니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 분할
X = df[['재료 및 마감 품질', '지상 거실 평방피트', '총 지하 평방피트']]
y = df['부동산 판매 가격']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```
```python
Mean Squared Error: 1667657527.1633682
```

## 4.1 랜덤 포레스트 회귀 모델링

* 랜덤 포레스트는 결정 트리(Decision Tree)를 기반으로 하는 앙상블 기법 중 하나이다.
* 여러 개의 결정 트리들을 학습시키고, 그 결과를 종합하여 예측한다
* 회귀와 분류 둘 다에 사용할 수 있는 방법이며, 해당 코드에서는 회귀 문제를 해결하기 위한 RandomForestRegressor를 사용해 봤습니다.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer

# 훈련 데이터를 로드합니다.
df = pd.read_csv('data_normalized.csv')

# SalePrice 컬럼을 따로 저장합니다.
y = df['SalePrice']
df_without_target = df.drop('SalePrice', axis=1)

# NaN 값을 평균값으로 대체합니다.
imputer = SimpleImputer(strategy='mean')
df_without_target = pd.DataFrame(imputer.fit_transform(df_without_target), columns=df_without_target.columns)

# 특성 선택
X_train, X_test, y_train, y_test = train_test_split(df_without_target, y, test_size=0.2)

selector = SelectKBest(score_func=chi2, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 특성 상관관계 축소
corr = df_without_target.corr()
high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = high_corr.unstack().reset_index()
high_corr = high_corr.loc[high_corr['level_0'] != high_corr['level_1'], :]

# 특성 제거
drop_cols = high_corr.loc[high_corr[0] > 0.9, 'level_0']
df_without_target = df_without_target.drop(columns=drop_cols)

# 특성 정규화
scaler = StandardScaler()
df_without_target = scaler.fit_transform(df_without_target)

# 훈련 데이터와 테스트 데이터로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(df_without_target, y, test_size=0.2)

# 모델 학습
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print(f'R^2: {model.score(X_test, y_test):.2f}')

# 결과

R^2: 0.92
```



# 5.모델 검증

선택된 모델의 성능을 평가하기 위해 교차 검증을 사용합니다. k-겹 교차 검증을 통해 모델의 일반화 성능을 추정할 수 있습니다.

* 교차 검증: 모델의 안정성을 확인하기 위해 교차 검증을 수행합니다.
* 성능 지표: RMSE, R^2 등의 성능 지표를 사용하여 모델의 예측 성능을 평가합니다.
  

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# 모델 정의
model = RandomForestRegressor(n_estimators=100)

# RMSE를 계산하기 위한 스코어 함수 정의
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# k-겹 교차 검증을 사용하여 성능 지표 계산
# 일반적으로 k=10으로 설정하지만, 다른 값도 사용 가능
k = 10

# RMSE 교차 검증 점수
rmse_scores = cross_val_score(model, df_without_target, y, cv=k, scoring=rmse_scorer)
# R^2 교차 검증 점수
r2_scores = cross_val_score(model, df_without_target, y, cv=k, scoring="r2")

print(f"RMSE (k-fold CV): {-rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
print(f"R^2 (k-fold CV): {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")

```
OutPut

```python
RMSE (k-fold CV): 28693.94 ± 6075.55
R^2 (k-fold CV): 0.86 ± 0.04
```

#### RMSE:값: 28693.94 ± 6075.55
* 설명: 모델의 예측 오차를 나타내는 지표. 평균적으로 예측이 실제 값과 약 28,693.94만큼 차이나며, 교차 검증에서의 변동은 ±6075.55 이다.

#### R&2(결정계수):값: 0.86 ± 0.04
* 설명: 모델이 데이터의 86%의 분산을 설명하고 있으며, 교차 검증에서의 변동은 ±0.04으로 나타났다.

#### 종합 평가:
* 모델의 설명력은 높으나 RMSE 값으로 볼 때 실제 값과의 차이도 존재합니다. 교차 검증의 표준편차를 통해 모델의 성능이 꽤 안정적임을 확인할 수 있습니다.

* 변수 선택: 피어슨 상관 계수는 선형 관계를 측정합니다. 가장 높은 값을 가진 세 변수는 목표 변수와의 선형 관계가 강하므로 선택되었다.

* 선형 회귀: 선택된 변수들은 부동산 판매 가격과의 선형 관계를 가지므로 선형 회귀가 적합하다 판단 되었다.

* 랜덤 포레스트 회귀 분류 : 회귀와 분류 둘 다에 사용할 수 있는 방법이며, 해당 코드에서는 회귀 문제를 해결하기 위한 RandomForestRegressor를 사용해 봤습니다.

* 모델 검증: 모델의 일반화 성능을 추정하기 위해 교차 검증을 사용합니다. 여러 훈련 및 테스트 세트에 걸쳐 모델을 평가하여 더 안정된 성능 지표를 얻을 수 있었다.

# 6.결과 해석 및 고찰

* 재료 및 마감 품질: 이 변수의 피어슨 상관 계수가 가장 높았습니다(0.7909816). 이는 '재료 및 마감 품질'이 부동산 판매 가격에 가장 큰 영향을 미친다는 것을 의미합니다. 높은 품질의 재료와 마감은 부동산의 가격을 크게 높일 수 있습니다.

* 지상 거실 평방피트: 이 변수는 부동산 판매 가격과 상당히 높은 상관 관계(0.7086245)를 가집니다. 크고 넓은 거실은 주택의 편안함과 활용성을 높이므로 가격에 긍정적인 영향을 미칠 수 있습니다.

* 총 지하 평방피트: 이 변수와 부동산 판매 가격의 상관 관계는 0.6135806으로, 다른 두 변수에 비해 상대적으로 낮지만 여전히 중요한 요소입니다. 넓은 지하 공간은 추가적인 활용 가능성(예: 저장 공간, 게임 룸, 워크샵 등)을 제공하므로 가격에 긍정적인 영향을 미칠 수 있습니다.

## 고찰:

* 선형 회귀 모델의 계수를 확인하여 각 변수의 영향력의 크기와 방향성을 구체적으로 알 수 있었습니다. 예를 들면, 모델의 계수가 양수라면 해당 변수는 부동산 가격에 긍정적인 영향을 미치는 것으로 해석되며, 그 크기는 해당 계수의 절대값에 따라 달라집니다.

# 7.결론 도출 및 추천

## 결론:

* '재료 및 마감 품질'은 부동산 판매 가격에 가장 큰 영향을 미치는 요소였다. 따라서 투자자나 건축업자는 재료 및 마감의 품질 향상에 주시해야 한다.
* '지상 거실 평방피트'와 '총 지하 평방피트' 역시 부동산 가격에 중요한 영향을 미치는데, 넓은 생활 공간은 부동산의 가치를 높이는 데 기여할 수 있다.

### 한계점

* 데이터 범위: 분석에 사용된 데이터는 특정 기간 또는 지역에 국한되어 있다.이로 인해 다른 시기나 지역에 적용할 때 결과가 달라질 수 있다.
* 이상치 처리: 사분위수 방법을 사용하여 이상치를 제거했지만, 이 방법이 항상 최적의 방법은 아닐 수 있다. 다른 방법도 생각해 볼 필요가 있다.

### 보완점

* 데이터 확장: 다른 기간이나 지역의 데이터를 추가하여 모델의 일반화 능력 향상 가능하다.
* 변수 추가: 부동산 가격에 영향을 미치는 다양한 변수들을 추가하여 분석의 정확성을 높일 수 있다.
* 다양한 모델 적용: 선형 회귀 외에도 다양한 머신 러닝 모델(랜덤 포레스트, 그래디언트 부스팅, 신경망 등)을 적용하여 최적의 모델을 찾아볼 수 있다.
* 이상치 처리 방법 개선: 다양한 이상치 탐지 및 처리 방법을 실험하여 데이터의 특성에 가장 적합한 방법을 찾아보는 것이 좋다.
* 모델 검증: 교차 검증(cross-validation)을 활용하여 모델의 안정성과 일반화 능력을 더욱 검증할 수 있다.







