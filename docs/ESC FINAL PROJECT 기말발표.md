# ESC FINAL PROJECT 기말발표

주제: **Linear Regression**을 통한 아파트 경매 가격 상승률 예측

**1조** **강태환** **김동휘** **이재현** **조유림** **최익준**



## 1. EDA

### 1) 중간발표 중요 내용

- 날짜 관련 파생변수 생성 : 최종 경매일과 최초 경매일 차이

- 시군구/감정사 클러스터링 진행 후 더미변수 생성
- Y와의 correlation이 높으면서 서로 correlation이 높은 6개의 변수 $\rightarrow$ PCA를 통해 차원 축소

### 2) 변경 사항

1. floor(층수) $\rightarrow$ skyscraper 더미변수 생성

30층 이상의 고층건물은 1, 30층 미만은 0의 값을 부여해 더미 변수 생성 

2. Hammer price/ minimum sales price(최저 매각 가격) Y값 새롭게 생성하는 방식 $\rightarrow$ **기존의 Y값**인 Hammer price를 예측하기로 결정

변환한 y값에 대해서 OLS 모형의 설명력이 너무 낮았음

3. Y값에 대해 **BOX-COX Transformation** 진행

```
plt.hist(data['Hammer_price'])
plt.show()
```

<img src="https://i.imgur.com/32qihgZ.png" style="zoom:80%;" />

기존의 hammer price 분포는 정규분포 형태가 아니다. OLS 가정사항 위반!

```
lambda_boxcox = boxcox(data['Hammer_price'])[1]
data['Hammer_price'] = boxcox(data['Hammer_price'])[0]
```

```
plt.hist(data['Hammer_price'])
plt.show()
```

<img src="https://i.imgur.com/1IQWUwL.png" style="zoom:80%;" />

BOX COX Transformation을 통해 정규분포 형태로 만들어줌

## 2. 모델링 

### 1) Frequentist 관점 OLS

**backward selection**을 통해 변수 선택

> 단계1 : 변수 전체 사용

```
from statsmodels.formula.api import ols
```

```
res = ols('Hammer_price ~ PC1 + PC2 + Claim_price + Auction_count + Final_First_auction_data + ad_si_0 + ad_si_1 + ad_si_2 + Appr_0 + Appr_1 + 서울 + skyscraper',data = data).fit()

res.summary()
```

<img src="https://i.imgur.com/7OMlt4v.png" style="zoom:80%;" />

- Claim price의 p-value가 0.319로 제거

> 단계2 : Claim price 제거 후 fitting

```
res2 = ols('Hammer_price ~ PC1 + PC2 + Auction_count + Final_First_auction_data + ad_si_0 + ad_si_1 + ad_si_2 + Appr_0 + Appr_1 + 서울 + skyscraper',data = data).fit()

res2.summary()
```

<img src="https://i.imgur.com/k4gmt5N.png" style="zoom:80%;" />

- Appr_0 변수의 p-value값이 0.2로 유의하지 않음
- Appr_0는 감정사 변수에 대한 dummy variable로 Appr_1 변수도 함께 제거해주기로 함

> 단계3 : 최종모형

```
res3 = ols('Hammer_price ~ PC1 + PC2 + Auction_count + Final_First_auction_data + ad_si_0 + ad_si_1 + ad_si_2 + 서울 + skyscraper', data = data).fit()

res3.summary()
```

<img src="https://i.imgur.com/U4H1wHQ.png" style="zoom:80%;" />

- 모든 변수가 유의하다고 나옴
- 최종 변수 : PC1, PC2, Auction_count(총 경매횟수), Final_First_auction_data(최종 경매일과 최초 경매일 차이), 시군구 더미변수, 시도 더미변수,  고층건물 더미변수
- adjusted r square값은 0.866

### 2) Bayesian Linear Regression

>  **단계 1 . Model Selection**

```
class Model:
    
    def z_function(self,data,category_index):
        p = data.shape[1]
        category = [i for i in range(p)]
        Ncategory = []
        for i in category_index:
            Ncategory.append(i)
            for j in i:
                category.remove(j)
        x = [[i] for i in category]
        for i in Ncategory:
            x.append(i)
        result = []
        cnt=0
        for i in range(len(x)): 
            count = len(list(combinations(x,i+1))) 
            for j in range(count):   
                z = [0]*p
                for k in range(i+1):
                    if len(list(combinations(x,i+1))[j][k]) == 1: #(1,2)
                        z[list(combinations(x,i+1))[j][k][0]] = 1
                    else:
                        for m in range(len(list(combinations(x,i+1))[j][k])):#(1,(2,3,4))
                            z[list(combinations(x,i+1))[j][k][m]] = 1
                result.append(z)
        return result
    
    def sig(self,X,y):
        X = np.array(X)
        n = X.shape[0]
        yhat = X@inv(t(X)@X)@t(X)@y
        res = y-yhat
        return sum(res**2)/n
        
    
    def posterior(self,X,y,category_index,g):
        nu0 = 1 
        z = Model.z_function(self,X,category_index)
        y = np.array(y)
        l = []
        for i in z:
            i = np.array(i)
            Xz = X.iloc[:,np.where(i==1)[0]]
            n,p = Xz.shape
            Xz = np.array(Xz)
            sig0 = Model.sig(self,Xz,y)
            ssr = t(y)@(np.eye(n)-g/(g+1)*Xz@inv(t(Xz)@Xz)@t(Xz))@y
            loglikelihood = (-n/2)*log(np.pi)+loggamma((nu0+n)/2)-loggamma(nu0/2)-p/2*log(1+g)+nu0/2*log(nu0*sig0)-(nu0+n)/2*log(nu0*sig0+ssr)
            l.append(loglikelihood)
        l = l/sum(l)
        return l
```

**z_function(self,data,category_index)**

- PC1 + PC2 + Claim_price + Auction_count + Final_First_auction_data + ad_si_0 + ad_si_1 + ad_si_2 + Appr_0 + Appr_1 + 서울 + skyscraper 변수 포함여부를 z 로 나타냄
- 더미 변수에 대해서는 제거할 때 동시에 제거하고 포함할 때는 동시에 포함하도록 설계

```
z = m.z_function(X,[[5,6,7],[8,9]]) # 카테고리 변수의 인덱스를 입력할 수 있도록 함
```

```
[[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
 [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 ...
```

- 511 경우의 수!!

**posterior(self,X,y,category_index,g)**

- weak prior 부여

$\rightarrow g=n,\ \nu_0=1, \sigma^2_0 = \text{estimated residual variance under the least squares estimate}$

$\rightarrow$ unit information prior for $p(\sigma^2)$

- log likelihood값을 이용해 posterior값을 구함

<img src="https://i.imgur.com/P0tXF4b.png" style="zoom:80%;" />

```
np.argmax(posterior)
509

result.iloc[509,:]
model        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

- PC1 변수를 제외하고 다른 변수 모두 사용하기로 결정



> **단계2 $\beta$ 샘플링**

```
from scipy.stats import gamma
from scipy.stats import multivariate_normal

BETA = []

for i in range(1000):
    precision_MC = gamma.rvs(a = nu0+n, scale = (nu0+n)/((nu0*s0)+ssr),size=1)
    cov_MC = 1/precision_MC * inv(t(Xz).dot(Xz))
    beta_MC = multivariate_normal.rvs(mean = beta * g/(g+1), cov=cov_MC*g/(g+1),size=1)
    BETA.append(beta_MC)
    
pd.DataFrame(BETA).mean()
```

- Monte Carlo approximation 을 이용해 beta 샘플링

$$
\begin{array}{l}
\text { sample } 1 / \sigma^{2} \sim \operatorname{gamma}\left(\left[\nu_{0}+n\right] / 2,\left[\nu_{0} \sigma_{0}^{2}+\mathrm{SSR}_{g}\right] / 2\right) ; \\
\text { sample } \boldsymbol{\beta} \sim \text { multivariate normal }\left(\frac{g}{g+1} \hat{\boldsymbol{\beta}}_{\text {ols }}, \frac{g}{g+1} \sigma^{2}\left[\mathbf{X}^{T} \mathbf{X}\right]^{-1}\right) \text { . }
\end{array}
$$

```
0     87.368541 #intercept
1     -1.632595 #PC2
2      0.000023 #Claim price
3     -2.945322 #Auction count
4      0.005012 #Final_First_auction_data
5     24.100314 #ad_si_0
6      9.487440 #ad_si_1
7     30.645763 #ad_si_2
8      7.222672 #Appr_0
9     29.947249 #Appr_1
10    14.348095 #서울
11     1.803751 #skyscrapper
```

- weak prior를 사용했기 때문에 일반 OLS에서 추정한 계수와 값이 유사한 것을 확인할 수 있음

<img src="https://i.imgur.com/eB1T8qO.png" style="zoom:80%;" />

## 3. Test data에 적용

Test data에 train data와 동일한 방식으로 전처리 진행

> 주요내용

1. 시군구/ 감정사

- train data에서 클러스터링 한 결과를 그대로 사용
- train data에 없는 감정사는 가장 큰 규모의 군집으로 넣어줌 

```
for i in range(829):
    if testdata['addr_si'][i] == '서초구':
        testdata['ad_si_0'][i] = 1
    elif testdata['addr_si'][i] == '용산구':
        testdata['ad_si_0'][i] = 1
    elif testdata['addr_si'][i] == '송파구':
        testdata['ad_si_0'][i] = 1
```

2. PCA

- train data에서 구한 loading 값을 이용해서 계산

<img src="https://i.imgur.com/uaWUHrj.png" style="zoom:80%;" />

```
PC2 = []
for i in range(820):
    PC2.append(PC11[i]+PC22[i]+PC33[i]+PC44[i]+PC55[i]+PC66[i])
```

3. Y값을 구한 후에 inverse-transformation 진행

```
prehammer2 = inv_boxcox(hammer2, lambda_boxcox)
```



> 결과 : RMSE 기준

![](https://i.imgur.com/oZ7HV4a.png)

- 일반회귀 : RMSE 238613007
- Bayesian Linear Regression : RMSE 180564322
- Bayesian Linear Regression의 결과가 더 좋은 것을 확인할 수 있음

