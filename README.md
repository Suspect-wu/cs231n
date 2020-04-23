### cs231n
Tip:

我们选取分母布局，都是基于分母的维度

主要是行向量为主，列向量须另外讨论，方法一样。

>标量对向量的链式求导法则

假设$z$ 为标量， $x_{1,m},y_{1,n}$ 为行向量。
$$
(\frac{\part z}{\part x})_{1,m} =  (\frac{\part z}{\part y})_{1,n}(\frac{\part y}{\part x})_{n,m}
$$

$$
\frac{\part z}{\part y_1}= \frac{\part z}{\part y_n}(\frac{\part y_n}{\part y_{n-1}} \frac{\part y_{n-1}}{\part y_{n-2}} ...\frac{\part y_2}{\part y_1})
$$



>标量对多个矩阵求导法则

当$A,X,B,Y$ 为矩阵时， $z$  为标量， 其中 $z = f(Y), Y=AX+B$ , 我们需要求出 $\frac{\part z}{\part X}$

直接给出结论， $\frac{\part z}{\part X}=A^T \frac{\part z}{\part Y}, \frac{\part z}{\part A}=\frac{\part z}{\part A}X^T$ 



> 向量对向量求导法则

均由定义证明，取分母布局。

假设，$Z_{1,m}=a_{1,n}W_{n,m}+b_{1,m}$ ,我们$(\frac{\part Z}{\part a})_{m,n}=W^T$

用定义法求解即可。

假设，$Z_{1,m}=\sigma(a_{1,n})W_{n,m}+b_{1,m}$ ,我们$(\frac{\part Z}{\part a})_{m,n}=[W^T]_{m,n} [diag(\sigma^{'}(a))]_{n,n}$



##### DNN

> 反向传播

损失函数：
$$
J(W,b,x,y)=\frac{1}{2}||a^L-y||_{2}^2
$$
其中$a^L,y$ 分别为维度为$n\_out$ 的向量

我们用梯度下降计算每层的$W,b$ .

假设输出层为$L$ 层： $a^L = \sigma(z^L)=\sigma (a^{L-1} W^L+b^L)$   ，$a^{L-1}$为L-1层的输出

让 $O= a^L -y$

于是 $J(W,b,x,y)=\frac{1}{2}||a^L-y||_{2}^2=\frac{1}{2}OO^T=\frac{1}{2}(a^L -y)(a^L -y)^T$

下面有：
$$
\frac{\part J(W,b,x,y)}{\part W^L} =\frac{\part OO^T}{\part W^L}=(a^{L-1})^T\frac{\part OO^T}{\part z^l}
$$
之前为标量对矩阵求导，通过化简，变为标量对向量求导。
$$
\frac{\part J(W,b,x,y)}{\part z^l} =\frac{\part O^TO}{\part O}(\frac{\part O}{\part a^L} \frac{\part a^L}{\part z^L} )
$$


其中 $\frac{\part a^L}{\part z^L} =diag(\sigma^{'}(z^L))$,  $\frac{\part O}{\part a^L}=E$,  $\frac{\part OO^T}{\part O}=2O$

综上所述：
$$
\frac{\part J(W,b,x,y)}{\part W^L}= (a^{L-1})^TO diag(\sigma^{'}(z^L))=[(a^{L-1})^TO]\odot\sigma^{'}(z^L)
$$
$\frac{\part J(W,b,x,y)}{\part W^L}=(a^{L-1})^T[(a^L -y)\odot\sigma^{'}(z^L)]$

$\frac{\part J(W,b,x,y)}{\part b^L}=(a^L -y)\odot \sigma^{'}(z^L)$

对于第l层的未激活输出$z^l$, 他的梯度可以为：

tip: 此时，$z^l$为向量，则可以这样链式求导
$$
\delta^l=\frac{\part J(W,b,x,y)}{\part z^l}=\frac{\part J(W,b,x,y)}{\part z^{l+1}}(\frac{\part z^{l+1}}{\part z^l})=\delta^{l+1}\frac{\part z^{l+1}}{\part z^l}
$$
那么：

tip：此时，$W^l$ 为矩阵，则不能链式求导，只能用结论：当$A,X,B,Y$ 为矩阵时， $z$  为标量， 其中 $z = f(Y), Y=AX+B$ , 我们需要求出 $\frac{\part z}{\part X}$直接给出结论， $\frac{\part z}{\part X}=A^T \frac{\part z}{\part Y}, \frac{\part z}{\part A}=\frac{\part z}{\part A}X^T$ 

其中：$J$ 可以看作 $f(z^l)$, 而 $z^l=a^lW^{l+1}+b^{l+1}$ 
$$
\frac{\part J(W,b,x,y)}{\part W^l}=(a^{l-1})^T \frac{\part J(W,b,x,y)}{\part z^l}=(a^{l-1})^T \delta^l
$$

$$
\frac{\part J(W,b,x,y)}{\part b^l}=\frac{\part J(W,b,x,y)}{\part z^l}=\delta^l
$$



另外：

$z^{l+1}=a^lW^{l+1}+b^{l+1}=\sigma(z^l)W^{l+1}+b^{l+1}$, 这个地方用到了向量对向量求导的准则。
$$
\frac{\part z^{l+1}}{\part z^l}=(W^{l+1})^Tdiag(\sigma^{'}(z^l))
$$
将上式带入$\delta^l$:
$$
\delta^l=\delta^{l+1}(W^{l+1})^Tdiag(\sigma^{'}(z^l))=[\delta^{l+1}(W^{l+1})^T]\odot\sigma^{'}(z^l)
$$



```python
def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None

    x_copy = np.reshape(x, (x.shape[0], -1))
    dw = x_copy.T @ dout
    dx = dout @ w.T
    dx = np.reshape(dx, x.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db
```



##### Softmax

> 反向传播

假设，$y,z$ 为行向量，$J(x,w,b,y)=-[\ln(softmax(z))]y^T=-\ln{\frac{e^z}{e^z 1^T}}y^T=-zy^T+1\ln(e^z1^T)y^T$

那么，我们可以得到：
$$
\frac{\part J}{\part z}=-y+\frac{\part 1\ln(e^z1^T)y^T}{\part e^z1^T}\frac{\part e^z1^T}{\part e^z}\frac{\part e^z}{\part z}=-y+(1\frac{1}{e^z1^T}y^T)1(diag(e^z))=-y+\frac{e^z}{e^z 1^T}=-y+a
$$


其中$1$ 为全部为1的行向量，$z$ 为未经过激活函数的输出行向量，$a$ 为经过激活函数的行向量。



##### RNN

> 前向传播

$h_t = \sigma(o^t)=\sigma ( x_t W_{xh}+h_{t-1}W_{hh}+b)$  其中$X_T$

序列号为 $t$ 的时候的输出为$z_t = h_tV+c$ , 然后我们的预测输出为 $a_t=\sigma(z^t)$, 输出为$Softmax$ 激活函数

> 反向传播误差

对与 RNN 而言，在每个位置都会有损失函数。

于是最终的损失函数为$L=\sum\limits_{t=1}^TL_t$

其中$V,C$ 的梯度计算较为简单

$$
\frac{\part L}{\part c}=\sum\limits_{t=1}^T \frac{\part{L_t}}{\part{c}}=\sum\limits_{t=1}^T \frac{\part{L_t}}{\part{z_t}}\frac{\part{z_t}}{\part{c}}=\sum\limits_{t=1}^T(a_t-y_t)
$$

$$
\frac{\part L}{\part V}=\sum\limits_{t=1}^T \frac{\part{L_t}}{\part{V}}=\sum\limits_{t=1}^T (h_t)^T\frac{\part{L_t}}{\part{z_t}}=\sum\limits_{t=1}^T(h_t)^T(a_t-y_t)
$$

在反向传播时，在某一序列位置t的梯度损失由**当前位置的输出对应的梯度损失**和**序列索引位置$t+1$时的梯度损失**两部分共同决定。

我们定义：
$$
\delta_t=\frac{\part L}{\part h_t}
$$

$$
\begin{align}
\delta_t&=\frac{\part L}{\part z_t}\frac{\part z_t}{\part h_t} +\frac{\part L}{\part h_{t+1}}\frac{\part h_{t+1}}{\part h_t}
\\&=(a_t-y_t)V^T+\delta_{t+1}\frac{\part h_{t+1}}{\part h_t}
\end{align}
$$

tip: $h_t = \sigma(o^t)=\sigma ( x_t W_{xh}+h_{t-1} W_{hh}+b)$ 此时的激活函数为$\tanh$ ,其导数为，
$$
\begin{align}
\frac{\part h_t}{\part h_{t-1}}=\frac{\part \sigma (h_{t-1}W_{hh})}{\part h_t}=(W_{hh})^Tdiag(\sigma^{'}(h_{t-1}))=(W_{hh})^Tdiag(1-h_{t-1}^2)
\end{align}
$$
那么：
$$
\delta_t=(a_t-y_t)V^T+\delta_{t+1}(W_{hh})^Tdiag(1-h_t^2)
\\ \delta_T=(a_T-y_T)V^T
$$

$$
\frac{\part L}{\part W_{hh}}=(h_{t-1})^T\frac{\part L}{\part o_t}=(h_{t-1})^T\frac{\part L}{\part h_t}\frac{\part h_t}{\part o_t}=(h_{t-1})^T\delta_tdiag(1-o_t^2)=(h_{t-1})^T[\delta_t\odot (1-o_t^2)]
$$

$$
\frac{\part L}{\part W_{xh}}=(x_t)^T\frac{\part L}{\part o_t}=(x_t)^T\frac{\part L}{\part h_t}\frac{\part h_t}{\part o_t}=(x_t)^T\delta_tdiag(1-o_t^2)=(x_t)^T[\delta_t\odot (1-o_t^2)]
$$

$$
\frac{\part L}{\part b}=\frac{\part L}{\part o_t}=\frac{\part L}{\part h_t}\frac{\part h_t}{\part o_t}=\delta_tdiag(1-o_t^2)=[\delta_t\odot (1-o_t^2)]
$$



```python
def rnn_step_backward(dnext_h, cache):

    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    x, Wx, Wh, prev_h, next_h, b = cache
    # dO = dnext_h * (1 - next_h ** 2)   
    # prev_h 为 h_{t-1}, next_h = h_t
    dWh = prev_h.T @ (dnext_h * (1- next_h ** 2))
    dWx = x.T @ (dnext_h * (1 - next_h ** 2))
    db = np.sum(dnext_h * (1 - next_h ** 2), axis=0)
    dx = (dnext_h * (1 - next_h ** 2)) @ Wh.T

    return dx, dprev_h, dWx, dWh, db

```



##### LSTM

> 门控机制

######  遗忘门

遗忘上一层隐藏细胞的状态。

$f^{t}=\sigma(h^{t-1}W_f+x^t U_f+b_f), f^t\in[0,1]$

$\sigma$ 多为sigmoid

$C^{t-1}\odot f^t$  作为上细胞的保留状态

###### 输入门

 控制当前序列位置的输入量

$i^t=\sigma(h^{t-1}W_i+x^tU_i+b_i)$

$a^i=\tanh(W_ah^{t-1}+U_ax^t+b_a)$

$\sigma$ 为sigmoid激活函数

$i^t \odot a^t$ 作为输入

###### 状态更新：

$C^t$ 为细胞状态，第一部分为上一细胞的遗忘量， 

$C^{t}=C^{t-1}\odot f^t+i^t \odot a^t$

###### 输出门

$o^t=\sigma(h^{t-1}W_o+x^t U_o+b_o)$

$h^t=o^t\odot \tanh(C^t)$



> 前向传播算法

1）更新遗忘门：

​                                                $f^{t}=\sigma(h^{t-1}W_f+x^{t}U_f+b_f)$

2) 更新输入门两部分输出：

​                                           $i^t=\sigma(h^{t-1}W_i+x^tU_i+b_i)$

​                                         $a^i=\tanh(h^{t-1}W_a+U_ax^t+b_a)$

3）更新细胞状态：

​                                          $C^{t}=C^{t-1}\odot f^t+i^t \odot a^t$

4）更新输出门输出：

​                                    $o^t=\sigma(h^{t-1}W_o+x^t U_o+b_o)$

​                                       $h^t=o^t\odot \tanh(C^t)$

5) 更新当前学列输出预测：

​                                        $\widehat{y}^t=\sigma(h^t V+b)$

> 反向传播算法

我们定义：
$$
\begin{align}
&\delta_h^t=\frac{\part L}{\part h^t}
\\&\delta_C^t=\frac{\part L}{\part C^t}
\end{align}
$$
对于末尾序列的那个位置的$\delta_h $ 与 $\delta_C$ 的导数：
$$
\begin{align}
&\delta_h^\tau=\frac{\part L}{\part h^\tau V+b}V^T=(\widehat{y}^\tau-y^\tau)V^T
\\&\delta_C^\tau=\frac{\part L}{\part C^\tau}=\frac{\part L}{\part h^\tau}\frac{\part h^\tau}{\part C^\tau}=\delta_h^\tau [diag(o^\tau) \odot diag(1-\tanh^2(C^\tau))]=\delta_h^\tau \odot o^\tau \odot(1-\tanh^2(C^\tau))
\end{align}
$$


$\delta_h^t$ 的梯度由本层t时刻的输出梯度误差和大于t时刻的误差两部分决定，对于内层相应的导数：
$$
\begin{align}
\delta_h^t=\frac{\part L}{\part h^t}=\frac{\part L}{\part h^t}+\frac{\part L}{\part h^{t+1}}\frac{\part h^{t+1}}{\part h^t}&=(\widehat{y}^t-y^t)V^T+\delta^{t+1}\frac{\part h^{t+1}}{\part h^t}
\\ &=(\widehat{y}^t-y^t)V^T+\delta^{t+1}\frac{\part h^{t+1}}{\part C^{t+1}}[\frac{\part C^{t+1}}{\part f^{t+1}}\frac{\part f^{t+1}}{\part h^t} 
\\ &  + \frac{\part C^{t+1}}{\part a^{t+1}}\frac{\part a^{t+1}}{\part h^t}+\frac{\part C^{t+1}}{\part i^{t+1}}\frac{\part i^{t+1}}{\part h^t} ] + \delta^{t+1} \frac{\part h^{t+1}}{\part o^{t+1}} \frac{\part o^{t+1}}{\part h^t}

\end{align}
$$
先来分析:
$$
\frac{\part h^{t+1}}{\part C^{t+1}}=diag(o^t\odot \tanh^{'}(C^{t+1}))
$$


下面我们来分析 
$$
\frac{\part C^{t+1}}{\part f^{t+1}}\frac{\part f^{t+1}}{\part h^t}  + \frac{\part C^{t+1}}{\part a^{t+1}}\frac{\part a^{t+1}}{\part h^t}+\frac{\part C^{t+1}}{\part i^{t+1}}\frac{\part i^{t+1}}{\part h^t}
$$

$$
\begin{align}
&
\frac{\part C^{t+1}}{\part f^{t+1}}\frac{\part f^{t+1}}{\part h^t}=diag(C^t\odot f^{t+1})diag(\sigma^{'})W_f^T=diag(C^t\odot f^{t+1} \odot \sigma^{'})W_f^T \\
& \frac{\part C^{t+1}}{\part a^{t+1}}\frac{\part a^{t+1}}{\part h^t}=diag(a^{t+1}\odot i^{t+1})diag(\sigma^{'})W_a^T=diag(a^{t+1}\odot i^{t+1} \odot \tanh^{'})W_a^T \\
& \frac{\part C^{t+1}}{\part i^{t+1}}\frac{\part i^{t+1}}{\part h^t}=diag(a^{t+1}\odot i^{t+1})diag(\sigma^{'})W_i^T=diag(a^{t+1}\odot i^{t+1} \odot \sigma^{'})W_i^T \\
\end{align}
$$

再来分析
$$
\frac{\part h^{t+1}}{\part o^{t+1}} \frac{\part o^{t+1}}{\part h^t}=diag(o^{t+1}\odot \tanh(C^{t+1})\odot \sigma^{'})W_o^T
$$
综上所述：
$$
\begin{align}
\delta_h^t & =(\widehat{y}^t-y^t)V^T+\delta^{t+1}\frac{\part h^{t+1}}{\part C^{t+1}}[diag(C^t\odot f^{t+1} \odot \sigma^{'})W_f^T+diag(a^{t+1}\odot i^{t+1} \odot \tanh^{'})W_a^T
\\& +diag(a^{t+1}\odot i^{t+1} \odot \sigma^{'})W_i^T]+\delta^{t+1}diag(o^{t+1}\odot \tanh(C^{t+1})\odot \sigma^{'})W_o^T
\end{align}
$$


下面，我们来分析$\delta_C^t$

而 $\delta_C^t$ 的反向梯度误差由前一层$\delta_C^{t+1}$的梯度误差和本层的从$h_t$传回来的梯度误差两部分组成，即：
$$
\begin{align}
\delta_C^t & =\frac{\part L}{\part C^{t+1}} \frac{\part C^{t+1}}{\part C^t}+\frac{\part L}{\part h^t} \frac{\part h^t}{\part C^t}\\
& =\delta_C^{t+1} diag(f^{t+1})+\delta_h^tdiag(o^t \odot\tanh^{'}(C^t))\\
& = \delta_C^{t+1}\odot f^{t+1} + \delta_h^t\odot o^t \odot\tanh^{'}(C^t)
\end{align}
$$
最后，我们就可更新梯度了呀！

我们仅求一个，其他的以此类推。
$$
\begin{align}
\frac{\part L}{\part W_f} & =\sum\limits_{i=1}^\tau((h^{t-1})^T\frac{\part L}{\part C_t}\frac{\part C^t}{\part f_t}\frac{\part f_t}{\part h^{t-1}W_f+x^{t}U_f+b_f})\\
& =\sum\limits_{i=1}^\tau((h^{t-1})^T\delta_C^t diag(C^{t-1})diag(\sigma^{'}) \\
& = \sum\limits_{i=1}^\tau((h^{t-1})^T\delta_C^t \odot C^{t-1} \odot \sigma^{'})
\end{align}
$$

tip 对$softmax$ 函数求导 :
$$
\frac{\part \frac{e^z}{e^z 1^T}}{\part z} =  \frac{1}{e^z 1^T}\frac{\part e^z}{\part z} +\frac{\part \frac{1}{e^z 1^T} e^z}{\part z} =\frac{1}{e^z 1^T}diag(e^z)-(\frac{\part \frac{1}{e^z 1^T} }{\part z})^T e^z=\frac{1}{e^z 1^T}diag(e^z)- \frac{1}{(e^z 1^T)^2}(e^z)^Te^z
$$



#### 1、KNN, SVM, Softmax，Network 

> SoftMax更新梯度

$l=-\log (SoftMax(X_{1,D}W_{D,K}))y_{K,1}$

$a=XW$

$l=-\log (SoftMax(a))y$

$dl=-d\log (SoftMax(a))y
=tr(-d\log (SoftMax(a))y)
=tr(-yd\log (SoftMax(a)))$

$d\log (SoftMax(a))=d\log \frac{\exp (a)}{\exp (a) 1_{k,1}}=da-(d\log \exp (a) 1_{k,1}) 1^{T}$

$dl=tr(-yda+y(d\log \exp (a) 1_{k,1}) 1^{T}),tr(AB)=tr(BA)$

$dl=tr(-yda+ 1^T y(d\log \exp (a) 1_{k,1})), 1^{T}y=1 $

$dl=tr(-yda+(d\log \exp (a)  1_{k,1}))$ 

$d\log \exp (a) 1_{k,1} = \frac{d \exp (a) 1_{k,1}}{\exp (a) 1_{k,1}}$ 
$=\frac{\exp a_{1,k} \odot da_{1,k} 1_{k,1}}{\exp (a) 1_{k,1}}$

$=\frac{\exp a_{1,k}^T \odot 1_{k,1} da_{1,k}}{\exp (a) 1_{k,1}}=\frac{\exp a_{1,k}^T da_{1,k}}{\exp (a) 1_{k,1}} $

$dl=tr(-yda+(d\log \exp a 1_{k,1}))=tr(-yda+\frac{\exp a_{1,k}^T da_{1,k}}{\exp (a) 1_{k,1}})$

$=tr[(-y+\frac{\exp a_{1,k}^T }{\exp a 1_{k,1} da}]$

$\frac{\partial l}{\partial a}=-y^T+\frac{\exp a_{1,k}}{\exp a 1_{k,1}}=-y^T+SoftMax(a)$

$dl=tr[(-y+\frac{\exp a_{1,k}^T}{\exp (a) 1_{k,1}})da]=tr[(-y+\frac{\exp a_{1,k}^T}{\exp (a) 1_{k,1}})dXW] $

$=tr[(-y+\frac{\exp a_{1,k}^T}{\exp (a) 1_{k,1}})XdW]$

$\frac{\partial l}{\partial W}=X^T(-y^T+\frac{\exp a_{1,k}}{\exp (a) 1_{k,1}})$



> BP算法

![](https://github.com/Suspect-wu/cs231n/blob/master/杂乱无章/BP.png?raw=true)







