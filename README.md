# EDA_Profit_Expect

# 주택 가격 예측 - 고급 회귀 기법
* 판매 가격을 예측하고 기능 엔지니어링, RF 및 그래디언트 부스팅 연습

데이터셋 출처(https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

# 데이터 로드
```python
import pandas as pd
df = pd.read_csv('./train.csv')
df.head()
```
# 데이터 확인 및 전처리

```python

Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	RL	65.0	8450	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500
1	2	20	RL	80.0	9600	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500
2	3	60	RL	68.0	11250	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500
3	4	70	RL	60.0	9550	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000
4	5	60	RL	84.0	14260	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000
5 rows × 81 columns
```

- 행과열 :1460 rows × 81 columns
- Null, Nan 값이 다수로 보입니다.
- 열이 전부 영어로 돼 있습니다.



### 모든 열을 한국어로 재구성

```python
df.columns = ['Id', '건물 클래스', '일반구역 분류', '건물 연결 거리', '부지 크기', '도로 접근 유형', '골목 진입 방식', ...
              '속성 모양', '평탄도', '유틸리티 유형', '로트 구성', '속성 기울기', '인근 지역', '주요 도로/철도 인접1', 
              ...
              '부동산 판매 가격']
```
### Null, Nan을 가진 열 제거
```python
df.dropna(inplace = True,axis = 1)
df.head()
```

```python

Id	건물 클래스	일반구역 분류	부지 크기	도로 접근 유형	속성 모양	평탄도	유틸리티 유형	로트 구성	속성 기울기	...	닫힌 현관 면적	3계절 베란다 면적	스크린 베란다 면적	풀 면적	기타 기능 가치	판매 월	판매 연도	판매 유형	판매 조건	부동산 판매 가격
0	1	60	RL	8450	Pave	Reg	Lvl	AllPub	Inside	Gtl	...	0	0	0	0	0	2	2008	WD	Normal	208500
1	2	20	RL	9600	Pave	Reg	Lvl	AllPub	FR2	Gtl	...	0	0	0	0	0	5	2007	WD	Normal	181500
2	3	60	RL	11250	Pave	IR1	Lvl	AllPub	Inside	Gtl	...	0	0	0	0	0	9	2008	WD	Normal	223500
3	4	70	RL	9550	Pave	IR1	Lvl	AllPub	Corner	Gtl	...	272	0	0	0	0	2	2006	WD	Abnorml	140000
4	5	60	RL	14260	Pave	IR1	Lvl	AllPub	FR2	Gtl	...	0	0	0	0	0	12	2008	WD	Normal	250000
5 rows × 62 columns
```

### 결과

5 rows x 81 columns ==> 5 rows x 62 columns 







