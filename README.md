### cs231n
头发越来越少
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







