### cs231n
头发越来越少
#### 1、KNN, SVM, Softmax，Network 

> SoftMax更新梯度

$$
\begin{align}
  & l=-\log (SoftMax({{X}_{1,D}}{{W}_{D,K}}))y_{K,1}^{{}} \\ 
 & a=XW \\ 
 & l=-\log (SoftMax(a))y \\ 
 & dl=-d\log (SoftMax(a))y \\ 
 & dl=tr(-d\log (SoftMax(a))y) \\ 
 & dl=tr(-yd\log (SoftMax(a))) \\ 
 & d\log (SoftMax(a))=d\log \frac{\exp (a)}{\exp (a){{1}_{k,1}}}=da-(d\log \exp (a){{1}_{k,1}}){{1}^{T}} \\ 
 & dl=tr(-yda+y(d\log \exp (a){{1}_{k,1}}){{1}^{T}}),tr(AB)=tr(BA) \\ 
 & dl=tr(-yda+{{1}^{T}}y(d\log \exp (a){{1}_{k,1}})),{{1}^{T}}y=1 \\ 
 & dl=tr(-yda+(d\log \exp (a){{1}_{k,1}})) \\ 
 & d\log \exp (a){{1}_{k,1}}=\frac{d\exp (a){{1}_{k,1}}}{\exp (a){{1}_{k,1}}}=\frac{(\exp {{(a)}_{1,k}}\odot d{{a}_{1,k}}){{1}_{k,1}}}{\exp (a){{1}_{k,1}}} \\ 
 & =\frac{(\exp {{(a)}_{1,k}}^{T}\odot {{1}_{k,1}})d{{a}_{1,k}}}{\exp (a){{1}_{k,1}}}=\frac{\exp {{(a)}_{1,k}}^{T}d{{a}_{1,k}}}{\exp (a){{1}_{k,1}}} \\ 
 & dl=tr(-yda+(d\log \exp (a){{1}_{k,1}}))=tr(-yda+\frac{\exp {{(a)}_{1,k}}^{T}d{{a}_{1,k}}}{\exp (a){{1}_{k,1}}}) \\ 
 & =tr[(-y+\frac{\exp {{(a)}_{1,k}}^{T}}{\exp (a){{1}_{k,1}}})da] \\ 
 & \frac{\partial l}{\partial a}=-{{y}^{T}}+\frac{\exp {{(a)}_{1,k}}^{{}}}{\exp (a){{1}_{k,1}}}=-{{y}^{T}}+SoftMax(a) \\ 
 & dl=tr[(-y+\frac{\exp {{(a)}_{1,k}}^{T}}{\exp (a){{1}_{k,1}}})da]=tr[(-y+\frac{\exp {{(a)}_{1,k}}^{T}}{\exp (a){{1}_{k,1}}})dXW] \\ 
 & =tr[(-y+\frac{\exp {{(a)}_{1,k}}^{T}}{\exp (a){{1}_{k,1}}})XdW] \\ 
 & \frac{\partial l}{\partial W}={{X}^{T}}(-{{y}^{T}}+\frac{\exp {{(a)}_{1,k}}^{{}}}{\exp (a){{1}_{k,1}}}) \\ 
 &  \\ 
\end{align}
$$



> BP算法

$$
\begin{align}
  & L=\text{-}\sum\limits_{i=1}^{n}{[\log (SoftMax(\sigma ({{X}_{i}}{{W}_{1}}+{{b}_{1}}){{W}_{2}}+{{b}_{2}})]{{y}_{i}}} \\ 
 & Input[{{X}_{i}},(1,D)]->FC[{{O}_{1,i}},()]->relu({{H}_{1,i}})->FC[{{O}_{2,i}},()]->SoftMax->output \\ 
 & {{O}_{2,i}}={{H}_{1,i}}{{W}_{2}}+{{b}_{2}}\text{----}{{H}_{1,i}}=\sigma ({{O}_{1,i}})\text{-------}{{O}_{1,i}}={{X}_{i}}{{W}_{1}}+b \\ 
 & dL=trd\text{ }\!\!\{\!\!\text{ -}\sum\limits_{i=1}^{n}{[\log (SoftMax({{O}_{2,i}})]{{y}_{i}}\text{ }\!\!\}\!\!\text{ }}=trd\text{ }\!\!\{\!\!\text{ -}\sum\limits_{i=1}^{n}{{{y}_{i}}[\log (\frac{\exp ({{O}_{2,i}})}{\exp ({{O}_{2,i}})1})]\text{ }\!\!\}\!\!\text{ }} \\ 
 & =trd\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}\{{{O}_{2,i}}-[\text{log}\exp ({{O}_{2,i}})1]{{1}^{T}}\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }} \\ 
 & dL=trd\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}{{O}_{2,i}}+{{y}_{i}}[\text{log}\exp ({{O}_{2,i}})1]{{1}^{T}}\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }}=trd\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}{{O}_{2,i}}+{{1}^{T}}{{y}_{i}}[\text{log}\exp ({{O}_{2,i}})1]\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }} \\ 
 & dL=trd\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}{{O}_{2,i}}+[\text{log}\exp ({{O}_{2,i}})1]\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}d{{O}_{2,i}}+d[\text{log}\exp ({{O}_{2,i}})1]\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }} \\ 
 & d[\text{log}\exp ({{O}_{2,i}})1]=\frac{d\exp ({{O}_{2,i}})1}{\exp ({{O}_{2,i}})1}=\frac{[\exp ({{O}_{2,i}})\odot d{{O}_{2,i}}]1}{\exp ({{O}_{2,i}})1}=\frac{\exp {{({{O}_{2,i}})}^{T}}\odot d{{O}_{2,i}}}{\exp ({{O}_{2,i}})1} \\ 
 & dL=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{-{{y}_{i}}d{{O}_{2,i}}+\frac{\exp {{({{O}_{2,i}})}^{T}}\odot d{{O}_{2,i}}}{\exp ({{O}_{2,i}})1}\text{ }\!\!\}\!\!\text{  }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{(-{{y}_{i}}+SoftMax{{({{O}_{2,i}})}^{T}}\text{)}d{{O}_{2,i}}\text{ }\!\!\}\!\!\text{ }} \\ 
 & \frac{\partial L}{\partial {{O}_{2,i}}_{{}}}=-{{y}_{i}}^{T}+SoftMax{{({{O}_{2,i}})}^{{}}} \\ 
 & d{{O}_{2,i}}=d({{H}_{1,i}}{{W}_{2}}+{{b}_{2}})=d{{H}_{1,i}}{{W}_{2}}+{{H}_{1,i}}d{{W}_{2}}+d{{b}_{2}} \\ 
 & dL=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}}^{T}}\text{(}d{{H}_{1,i}}{{W}_{2}}+{{H}_{1,i}}d{{W}_{2}}+d{{b}_{2}}\text{) }\!\!\}\!\!\text{ }} \\ 
 & \frac{\partial L}{\partial {{W}_{2}}}=\sum{{{H}_{1,i}}^{T}{{\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}}^{{}}}} \\ 
 & \frac{\partial L}{\partial {{b}_{2}}}=\sum{{{\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}}^{{}}}} \\ 
 & d{{L}_{2}}\text{=}tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}}^{T}}d{{H}_{1,i}}{{W}_{2}}\text{ }\!\!\}\!\!\text{ }}\text{=}tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{W}_{2}}{{\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}}^{T}}d{{H}_{1,i}}\text{ }\!\!\}\!\!\text{ }},d{{H}_{1,i}}=d\sigma ({{O}_{1,i}})={{\sigma }^{'}}({{O}_{1,i}})\odot d{{O}_{1,i}} \\ 
 & \frac{\partial L}{\partial {{H}_{1,i}}}=\frac{\partial L}{\partial {{O}_{2,i}}_{{}}}{{W}_{2}}^{T},d \\ 
 & d{{L}_{2}}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{H}_{1,i}}}}^{T}}d{{H}_{1,i}}\text{ }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{H}_{1,i}}}}^{T}}[{{\sigma }^{'}}({{O}_{1,i}})\odot d{{O}_{1,i}}\text{ }\!\!]\!\!\text{  }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{[\frac{\partial L}{\partial {{H}_{1,i}}}\odot {{\sigma }^{'}}({{O}_{1,i}})]}^{T}}d{{O}_{1,i}}\text{ }\!\!\}\!\!\text{ }} \\ 
 & \frac{\partial L}{\partial {{O}_{1,i}}}=\frac{\partial L}{\partial {{H}_{1,i}}}\odot {{\sigma }^{'}}({{O}_{1,i}}) \\ 
 & d{{L}_{2}}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{O}_{1,i}}}}^{T}}d{{O}_{1,i}}\text{ }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{O}_{1,i}}}}^{T}}d[{{X}_{i}}{{W}_{1}}+b]\text{ }\!\!\}\!\!\text{ }}=tr\text{ }\!\!\{\!\!\text{ }\sum\limits_{i=1}^{n}{{{\frac{\partial L}{\partial {{O}_{1,i}}}}^{T}}[{{X}_{i}}d{{W}_{1}}+db]\text{ }\!\!\}\!\!\text{ }} \\ 
 & \frac{\partial L}{\partial {{W}_{2}}}=\sum{{{X}_{i}}^{T}{{\frac{\partial L}{\partial {{O}_{1,i}}}}^{{}}}} \\ 
 & \frac{\partial L}{\partial {{b}_{2}}}=\sum{{{\frac{\partial L}{\partial {{O}_{1,i}}}}^{{}}}} \\ 
 &  \\ 
\end{align}
$$





$$a^y$$