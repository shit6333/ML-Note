###### tags: Machine learning
# 從 Linear Regression 到神經網路

## 前言
今天看了李宏毅老師 2021 機器學習的前兩集，真的受益良多
之前雖然玩過一些 ML 的應用，但對原理還是一知半解
這次算是理解了不少

這邊簡單做個筆記紀錄一下

[regression] => [Loss 和 Gradient decent] => [Activation] => [神經網路]

<br>

## Linear Regression
 linear regression 算是最基礎也最容易理解的演算法
 
 :::info
 $$y=w*x + b$$
 - $w$ : 權重(係數)
 - $b$ : bias，是可供調整的偏差
 - $x,y$ : 分別為資料的 input 和 output
 
  ![](https://i.imgur.com/1A7G49V.png =400x240)
  
  藍色點的是實際的資料
  此資料有兩個特徵為 x, y
  紅色線就是 linear regression，使我們能用 x 特徵預測 y特徵
  
  如果說 x,y有點抽象，想像這資料是一種"麵包"
  x 指麵包的"香味"，y 指麵包吃起來的"味道"
  而 regression 可以讓我們用香味來預測此麵包的味道
 :::

我們有回歸的基本知識了
那要如何找出最好的 regression 來做預測呢 ?
<br>
 
 ## Loss function
 要找出最好的線之前，我們首先要知道目前的 regression 離"最好"還多遠
 所以使用 loss function 來計算我們目前 regression 預測的好壞

評估的想法很簡單，就將我們**預測出的 y 與真實的 y 相減**就好啦
寫成  $error = y-\hat{y}$ 

由於我們 x 資料眾多，
所以我們還會利用其他方法將這些 error 變成一個值，做整體的評估
也就是 loss function 的用處，這裡介紹 MAE、MSE
:::info
### MAE (Mean Absolute Error)

$$L(w,b)=\frac{1}{n}\sum^n_i{|y_i-\hat{y}_i|}$$
- $w$ : 權重
- $b$ : bias
- $n$ : 資料數量
<br>

### MSE (Mean Square Error)
$$L(w,b)=\frac{1}{n}\sum^n_i{(y_i-\hat{y}_i)^2}$$

這兩種非常相似，要使用哪種就視情況而定
還有一種叫 Cross Entropy 常用在機率分布的資料評估上
:::
<br>

## Optimization 優化器
有了評估 regression 好壞的方法，這下可以來"優化"此函數了
我們優化 regression 的目標就是
**找出一組 (w,b) 可以讓 loss (error) 最小**
可以寫成 $w^*,b^* = argmin(L)$ -- loss 簡稱  L

優化器種類繁多
這裡介紹最基本的 "**梯度下降法**"

---
當我們將所有權重w 的 loss 列出來時
可以看到下圖
我們的目標就是到達有最小 loss 的 w

![](https://i.imgur.com/ohWyzGn.png)

:::info
### 梯度下降
梯度下降就是計算現在所在位置(w)的斜率
並利用此斜率 引導我們的 w 前往最低點
斜率可以利用對 loss 偏微分來獲得
$$斜率=\frac{\partial{L}}{\partial{w}}$$

有了斜率後，我們就可以跟著斜率慢慢下降
為了讓下降速度不要太快
會定義另一值，稱為 learning rate : $\eta$
以下為梯度下降公式
<br>

$$w^1 \leftarrow w^0 - \eta * \frac{\partial{L}}{\partial{w}}$$
- $w^0$ : 為初始權重
- $w^1$ : 為更新後權重

在更新權重後便會趨向 loss 低點了
**你可能會想問 bias(b) 呢 ?**
也是同樣方法，只是數值從 w 改成 b 了
:::

### local minimum vs global minimum
有時會碰到下降時卻卡在某區域的低點 (local minimum)
這時可以試著調整學習率，使下降速度上升些
也就是跨大步一點啦 ~
<br>

## Activation Function
我們知道如果利用 linear regression 可以預測資料
但當資料分布更加複雜，像以下這樣

![](https://i.imgur.com/hH6cCho.png =400x240)

很難光是用一個 $y=w*x+b$ 來做預測
這時我們可以用一種名為 Piecewise Linear 的東西拼湊出我們要的函數
也就是下圖**藍色線的相加**

![](https://i.imgur.com/qhP8TLe.png =400x240)

因為純 Piecewise Linear 在計算其函數時較不易
所以我們將它看作是 一個個的 sigmoid 的函數 (蠻像的吧~)

![](https://i.imgur.com/7F64wOa.png =400x240)

也就是說在使用這些 sigmoid 函數後，就可以更精準預測出資料
我們將此函數與原本 linear regression 結合
:::info
### sigmoid 函數
$$y = c_1 * \frac{1}{1+e^{-(w_1*x_1+b_1)}}=c_1*sigmoid(w_1*x_1+b_1)$$

<br>

同樣如更改 w,b 也會影響 sigmoid 

![](https://i.imgur.com/ZstNN00.png =500x340)

<br>

:::

有了此函數，之後再計算 loss 和 Update 權重方式基本都是相同的

也許你會想說，難道我不能用原來的 Piecewise Linear 嗎 ?
- 可以，有一種方式與 Piecewise Linear 相近
就叫做 ReLU，在之後也是超常用的函數

**那這些函數我們統稱叫 "Activation Funciton"**
<br>

## 神經網路
雖然標題暴雷了，但我要接續之前的 Activation function 講下去
當我們有許多"特徵"x 時 (ex : 香味、味道、外觀...)
我們的函數會將這些特徵加總
$y =b+ w_1x_1+w_2x_2+w_3x_3 ...=b+\sum_j{w_j*x_j}$

套用上一截的 Activation function 後變這樣
:::info
$$y=b+\sum_i{c_i*sigmoid(b_i+\sum_j{w_{ij}*x_j})}$$
:::
---
接著我們將其模組化，並使用矩陣運算
先取出最裡面的式子，設為 r
$r_i = b_i+\sum_j{w_{ij}*x_j}$

![](https://i.imgur.com/RMNw3BT.png =300x80)

$r=b+WX$
又可以拆解成下圖 (好像有點雛型了)

![](https://i.imgur.com/VnKlKNA.png =500x300)
<br>

之後我們將 Activation function 稱為 $\sigma$ 
並將輸出的值做 a
$a=\sigma{(r)}$

![](https://i.imgur.com/ihewn5P.png =500x300)
<br>

最後在乘上係數 $c_i$ 並加總輸出
(神經網路出現啦)

![](https://i.imgur.com/9ebq9x2.png =500x300)
<br>

如果我們 $a$ 不直接全部加總輸出，
還可在延伸更多層的網路 ~ **也就會變為 Deep Learning** !

![](https://i.imgur.com/TErqQcp.png =500x340)
<br>

---
最後我們會統稱上述提到的係數 (c,b,w...) 為 $\theta$
方便之後做計算
而之後優化也就如之前 linear regression 所說步驟一樣

- 先計算 loss
$L(\theta)$

- 計算 gradient (簡稱為 g)
$g=\bigtriangledown L(\theta^0)$

- 優化權重
$\theta^1 = \theta^0-g$

---

以上就是由 linear regression 到神經網路啦
看完老師教學真的受益頗深 ! 大推 !!

[李宏毅 機器學習2021](https://www.youtube.com/watch?v=Ye018rCVvOo&t=4s)