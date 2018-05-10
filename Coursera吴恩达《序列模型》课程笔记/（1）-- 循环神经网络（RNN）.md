> 我的CSDN博客地址：[红色石头的专栏](https://link.zhihu.com/?target=http%3A//blog.csdn.net/red_stone1)
> 我的知乎主页：[红色石头](https://www.zhihu.com/people/red_stone_wl)
> 我的微博：[RedstoneWill的微博](https://link.zhihu.com/?target=https%3A//weibo.com/6479023696/profile%3Ftopnav%3D1%26wvr%3D6%26is_all%3D1)
> 我的GitHub：[RedstoneWill的GitHub](https://link.zhihu.com/?target=https%3A//github.com/RedstoneWill)
> 我的微信公众号：红色石头的机器学习之路（ID：redstonewill）
> 欢迎大家关注我！共同学习，共同进步！

《Recurrent Neural Networks》是Andrw Ng深度学习专项课程中的第五门课，也是最后一门课。这门课主要介绍循环神经网络（RNN）的基本概念、模型和具体应用。该门课共有3周课时，所以我将分成3次笔记来总结，这是第一节笔记。

## 1\. Why sequence models

序列模型能够应用在许多领域，例如：

*   **语音识别**
*   **音乐发生器**
*   **情感分类**
*   **DNA序列分析**
*   **机器翻译**
*   **视频动作识别**
*   **命名实体识别**

![](https://pic3.zhimg.com/80/v2-77cab619738d3e9e45e8fc3397f624fd_hd.jpg)

这些序列模型基本都属于监督式学习，输入x和输出y不一定都是序列模型。如果都是序列模型的话，模型长度不一定完全一致。

## 2\. Notation

下面以命名实体识别为例，介绍序列模型的命名规则。示例语句为：

**Harry Potter and Hermione Granger invented a new spell.**

该句话包含9个单词，输出y即为1 x 9向量，每位表征对应单词是否为人名的一部分，1表示是，0表示否。很明显，该句话中“Harry”，“Potter”，“Hermione”，“Granger”均是人名成分，所以，对应的输出y可表示为：

![y=[1\ \ 1\ \ 0\ \ 1\ \ 1\ \ 0\ \ 0\ \ 0\ \ 0]](https://www.zhihu.com/equation?tex=y%3D%5B1%5C+%5C+1%5C+%5C+0%5C+%5C+1%5C+%5C+1%5C+%5C+0%5C+%5C+0%5C+%5C+0%5C+%5C+0%5D)

一般约定使用 ![y^{<t>}](https://www.zhihu.com/equation?tex=y%5E%7B%3Ct%3E%7D) 表示序列对应位置的输出，使用 ![T_y](https://www.zhihu.com/equation?tex=T_y) 表示输出序列长度，则 ![1\leq t\leq T_y](https://www.zhihu.com/equation?tex=1%5Cleq+t%5Cleq+T_y) 。

对于输入x，表示为：

![[x^{<1>}\ \ x^{<2>}\ \ x^{<3>}\ \ x^{<4>}\ \ x^{<5>}\ \ x^{<6>}\ \ x^{<7>}\ \ x^{<8>}\ \ x^{<9>}]](https://www.zhihu.com/equation?tex=%5Bx%5E%7B%3C1%3E%7D%5C+%5C+x%5E%7B%3C2%3E%7D%5C+%5C+x%5E%7B%3C3%3E%7D%5C+%5C+x%5E%7B%3C4%3E%7D%5C+%5C+x%5E%7B%3C5%3E%7D%5C+%5C+x%5E%7B%3C6%3E%7D%5C+%5C+x%5E%7B%3C7%3E%7D%5C+%5C+x%5E%7B%3C8%3E%7D%5C+%5C+x%5E%7B%3C9%3E%7D%5D)

同样， ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 表示序列对应位置的输入， ![T_x](https://www.zhihu.com/equation?tex=T_x) 表示输入序列长度。注意，此例中， ![T_x=T_y](https://www.zhihu.com/equation?tex=T_x%3DT_y) ，但是也存在 ![T_x\neq T_y](https://www.zhihu.com/equation?tex=T_x%5Cneq+T_y) 的情况。

如何来表示每个 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 呢？方法是首先建立一个词汇库vocabulary，尽可能包含更多的词汇。例如一个包含10000个词汇的词汇库为：

![\left[ \begin{matrix} a \\ and \\ \cdot \\ \cdot \\ \cdot \\ harry \\ \cdot \\ \cdot \\ \cdot \\ potter \\ \cdot \\ \cdot \\ \cdot \\ zulu \end{matrix} \right]](https://www.zhihu.com/equation?tex=%5Cleft%5B+%5Cbegin%7Bmatrix%7D+a+%5C%5C+and+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+harry+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+potter+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+%5Ccdot+%5C%5C+zulu+%5Cend%7Bmatrix%7D+%5Cright%5D)

该词汇库可看成是10000 x 1的向量。值得注意的是自然语言处理NLP实际应用中的词汇库可达百万级别的词汇量。

然后，使用one-hot编码，例句中的每个单词 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 都可以表示成10000 x 1的向量，词汇表中与 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 对应的位置为1，其它位置为0。该 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 为one-hot向量。值得一提的是如果出现词汇表之外的单词，可以使用UNK或其他字符串来表示。

对于多样本，以上序列模型对应的命名规则可表示为： ![X^{(i)<t>}](https://www.zhihu.com/equation?tex=X%5E%7B%28i%29%3Ct%3E%7D) ， ![y^{(i)<t>}](https://www.zhihu.com/equation?tex=y%5E%7B%28i%29%3Ct%3E%7D) ， ![T_x^{(i)}](https://www.zhihu.com/equation?tex=T_x%5E%7B%28i%29%7D) ， ![T_y^{(i)}](https://www.zhihu.com/equation?tex=T_y%5E%7B%28i%29%7D) 。其中， ![i](https://www.zhihu.com/equation?tex=i) 表示第i个样本。不同样本的 ![T_x^{(i)}](https://www.zhihu.com/equation?tex=T_x%5E%7B%28i%29%7D) 或 ![T_y^{(i)}](https://www.zhihu.com/equation?tex=T_y%5E%7B%28i%29%7D) 都有可能不同。

## 3\. Recurrent Neural Network Model

对于序列模型，如果使用标准的神经网络，其模型结构如下：

![](https://pic4.zhimg.com/80/v2-bdd092d23fc2a13c9e058992e8f5ea9b_hd.jpg)

使用标准的神经网络模型存在两个问题：

第一个问题，不同样本的输入序列长度或输出序列长度不同，即 ![T_x^{(i)}\neq T_x^{(j)}](https://www.zhihu.com/equation?tex=T_x%5E%7B%28i%29%7D%5Cneq+T_x%5E%7B%28j%29%7D) ， ![T_y^{(i)}\neq T_y^{(j)}](https://www.zhihu.com/equation?tex=T_y%5E%7B%28i%29%7D%5Cneq+T_y%5E%7B%28j%29%7D) ，造成模型难以统一。解决办法之一是设定一个最大序列长度，对每个输入和输出序列补零并统一到最大长度。但是这种做法实际效果并不理想。

第二个问题，也是主要问题，这种标准神经网络结构无法共享序列不同 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 之间的特征。例如，如果某个 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 即“Harry”是人名成分，那么句子其它位置出现了“Harry”，也很可能也是人名。这是共享特征的结果，如同CNN网络特点一样。但是，上图所示的网络不具备共享特征的能力。值得一提的是，共享特征还有助于减少神经网络中的参数数量，一定程度上减小了模型的计算复杂度。例如上图所示的标准神经网络，假设每个 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 扩展到最大序列长度为100，且词汇表长度为10000，则输入层就已经包含了100 x 10000个神经元了，权重参数很多，运算量将是庞大的。

标准的神经网络不适合解决序列模型问题，而循环神经网络（RNN）是专门用来解决序列模型问题的。RNN模型结构如下：

![](https://pic1.zhimg.com/80/v2-43c44de93c84599847dffdb0fd14f2bb_hd.jpg)

序列模型从左到右，依次传递，此例中， ![T_x=T_y](https://www.zhihu.com/equation?tex=T_x%3DT_y) 。 ![x^{<t>}](https://www.zhihu.com/equation?tex=x%5E%7B%3Ct%3E%7D) 到 ![\hat y^{<t>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D) 之间是隐藏神经元。 ![a^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D) 会传入到第 ![t+1](https://www.zhihu.com/equation?tex=t%2B1) 个元素中，作为输入。其中， ![a^{<0>}](https://www.zhihu.com/equation?tex=a%5E%7B%3C0%3E%7D) 一般为零向量。

RNN模型包含三类权重系数，分别是 ![W_{ax}](https://www.zhihu.com/equation?tex=W_%7Bax%7D) ， ![W_{aa}](https://www.zhihu.com/equation?tex=W_%7Baa%7D) ， ![W_{ya}](https://www.zhihu.com/equation?tex=W_%7Bya%7D) 。且不同元素之间同一位置共享同一权重系数。

![](https://pic3.zhimg.com/80/v2-f573e2cbb5f7e8638a25409661a04859_hd.jpg)

RNN的正向传播（Forward Propagation）过程为：

![a^{<t>}=g(W_{aa}\cdot a^{<t-1>}+W_{ax}\cdot x^{<t>}+ba)](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3Dg%28W_%7Baa%7D%5Ccdot+a%5E%7B%3Ct-1%3E%7D%2BW_%7Bax%7D%5Ccdot+x%5E%7B%3Ct%3E%7D%2Bba%29)

![\hat y^{<t>}=g(W_{ya}\cdot a^{<t>}+b_y)](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D%3Dg%28W_%7Bya%7D%5Ccdot+a%5E%7B%3Ct%3E%7D%2Bb_y%29)

其中， ![g(\cdot)](https://www.zhihu.com/equation?tex=g%28%5Ccdot%29) 表示激活函数，不同的问题需要使用不同的激活函数。

为了简化表达式，可以对 ![a^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D) 项进行整合：

![W_{aa}\cdot a^{<t-1>}+W_{ax}\cdot x^{<t>}=[W_{aa}\ \ W_{ax}]\left[ \begin{matrix} a^{<t-1>} \\ x^{<t>} \end{matrix} \right]\rightarrow W_a[a^{<t-1>},x^{<t>}]](https://www.zhihu.com/equation?tex=W_%7Baa%7D%5Ccdot+a%5E%7B%3Ct-1%3E%7D%2BW_%7Bax%7D%5Ccdot+x%5E%7B%3Ct%3E%7D%3D%5BW_%7Baa%7D%5C+%5C+W_%7Bax%7D%5D%5Cleft%5B+%5Cbegin%7Bmatrix%7D+a%5E%7B%3Ct-1%3E%7D+%5C%5C+x%5E%7B%3Ct%3E%7D+%5Cend%7Bmatrix%7D+%5Cright%5D%5Crightarrow+W_a%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D)

则正向传播可表示为：

![a^{<t>}=g(W_a[a^{<t-1>},x^{<t>}]+b_a)](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3Dg%28W_a%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_a%29)

![\hat y^{<t>}=g(W_{y}\cdot a^{<t>}+b_y)](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D%3Dg%28W_%7By%7D%5Ccdot+a%5E%7B%3Ct%3E%7D%2Bb_y%29)

值得一提的是，以上所述的RNN为单向RNN，即按照从左到右顺序，单向进行， ![\hat y^{<t>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D) 只与左边的元素有关。但是，有时候 ![\hat y^{<t>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D) 也可能与右边元素有关。例如下面两个句子中，单凭前三个单词，无法确定“Teddy”是否为人名，必须根据右边单词进行判断。

**He said, “Teddy Roosevelt was a great President.”**

**He said, “Teddy bears are on sale!”**

因此，有另外一种RNN结构是双向RNN，简称为BRNN。 ![\hat y^{<t>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D) 与左右元素均有关系，我们之后再详细介绍。

## 4\. Backpropagation through time

针对上面识别人名的例子，经过RNN正向传播，单个元素的Loss function为：

![L^{<t>}(\hat y^{<t>},y^{<t>})=-y^{<t>}log\ \hat y^{<t>}-(1-y^{<t>})log\ (1-\hat y^{<t>})](https://www.zhihu.com/equation?tex=L%5E%7B%3Ct%3E%7D%28%5Chat+y%5E%7B%3Ct%3E%7D%2Cy%5E%7B%3Ct%3E%7D%29%3D-y%5E%7B%3Ct%3E%7Dlog%5C+%5Chat+y%5E%7B%3Ct%3E%7D-%281-y%5E%7B%3Ct%3E%7D%29log%5C+%281-%5Chat+y%5E%7B%3Ct%3E%7D%29)

该样本所有元素的Loss function为：

![L(\hat y,y)=\sum_{t=1}^{T_y}L^{<t>}(\hat y^{<t>},y^{<t>})](https://www.zhihu.com/equation?tex=L%28%5Chat+y%2Cy%29%3D%5Csum_%7Bt%3D1%7D%5E%7BT_y%7DL%5E%7B%3Ct%3E%7D%28%5Chat+y%5E%7B%3Ct%3E%7D%2Cy%5E%7B%3Ct%3E%7D%29)

然后，反向传播（Backpropagation）过程就是从右到左分别计算 ![L(\hat y,y)](https://www.zhihu.com/equation?tex=L%28%5Chat+y%2Cy%29) 对参数 ![W_{a}](https://www.zhihu.com/equation?tex=W_%7Ba%7D) ， ![W_{y}](https://www.zhihu.com/equation?tex=W_%7By%7D) ， ![b_a](https://www.zhihu.com/equation?tex=b_a) ， ![b_y](https://www.zhihu.com/equation?tex=b_y) 的偏导数。思路与做法与标准的神经网络是一样的。一般可以通过成熟的深度学习框架自动求导，例如PyTorch、Tensorflow等。这种从右到左的求导过程被称为Backpropagation through time。

## 5\. Different types of RNNs

以上介绍的例子中， ![T_x=T_y](https://www.zhihu.com/equation?tex=T_x%3DT_y) 。但是在很多RNN模型中， ![T_x](https://www.zhihu.com/equation?tex=T_x) 是不等于 ![T_y](https://www.zhihu.com/equation?tex=T_y) 的。例如第1节介绍的许多模型都是 ![T_x\neq T_y](https://www.zhihu.com/equation?tex=T_x%5Cneq+T_y) 。根据 ![T_x](https://www.zhihu.com/equation?tex=T_x) 与 ![T_y](https://www.zhihu.com/equation?tex=T_y) 的关系，RNN模型包含以下几个类型：

*   **Many to many:** ![T_x=T_y](https://www.zhihu.com/equation?tex=T_x%3DT_y)
*   **Many to many:** ![T_x\neq T_y](https://www.zhihu.com/equation?tex=T_x%5Cneq+T_y)
*   **Many to one:** ![T_x>1,T_y=1](https://www.zhihu.com/equation?tex=T_x%3E1%2CT_y%3D1)
*   **One to many:** ![T_x=1,T_y>1](https://www.zhihu.com/equation?tex=T_x%3D1%2CT_y%3E1)
*   **One to one:** ![T_x=1,T_y=1](https://www.zhihu.com/equation?tex=T_x%3D1%2CT_y%3D1)

不同类型相应的示例结构如下：

![](https://pic1.zhimg.com/80/v2-ceb8134824502eeeb9c4f985b05d1a3b_hd.jpg)

## 6\. Language model and sequence generation

语言模型是自然语言处理（NLP）中最基本和最重要的任务之一。使用RNN能够很好地建立需要的不同语言风格的语言模型。

什么是语言模型呢？举个例子，在语音识别中，某句语音有两种翻译：

*   **The apple and pair salad.**
*   **The apple and pear salad.**

很明显，第二句话更有可能是正确的翻译。语言模型实际上会计算出这两句话各自的出现概率。比如第一句话概率为 ![10^{-13}](https://www.zhihu.com/equation?tex=10%5E%7B-13%7D) ，第二句话概率为 ![10^{-10}](https://www.zhihu.com/equation?tex=10%5E%7B-10%7D) 。也就是说，利用语言模型得到各自语句的概率，选择概率最大的语句作为正确的翻译。概率计算的表达式为：

![P(y^{<1>},y^{<2>},\cdots,y^{<T_y>})](https://www.zhihu.com/equation?tex=P%28y%5E%7B%3C1%3E%7D%2Cy%5E%7B%3C2%3E%7D%2C%5Ccdots%2Cy%5E%7B%3CT_y%3E%7D%29)

如何使用RNN构建语言模型？首先，我们需要一个足够大的训练集，训练集由大量的单词语句语料库（corpus）构成。然后，对corpus的每句话进行切分词（tokenize）。做法就跟第2节介绍的一样，建立vocabulary，对每个单词进行one-hot编码。例如下面这句话：

**The Egyptian Mau is a bread of cat.**

One-hot编码已经介绍过了，不再赘述。还需注意的是，每句话结束末尾，需要加上**< EOS >**作为语句结束符。另外，若语句中有词汇表中没有的单词，用**< UNK >**表示。假设单词“Mau”不在词汇表中，则上面这句话可表示为：

**The Egyptian < UNK > is a bread of cat. < EOS >**

准备好训练集并对语料库进行切分词等处理之后，接下来构建相应的RNN模型。

![](https://pic4.zhimg.com/80/v2-f2f5643b1c850e900d96c8f6dc5fe36f_hd.jpg)

语言模型的RNN结构如上图所示， ![x^{<1>}](https://www.zhihu.com/equation?tex=x%5E%7B%3C1%3E%7D) 和 ![a^{<0>}](https://www.zhihu.com/equation?tex=a%5E%7B%3C0%3E%7D) 均为零向量。Softmax输出层 ![\hat y^{<1>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3C1%3E%7D) 表示出现该语句第一个单词的概率，softmax输出层 ![\hat y^{<2>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3C2%3E%7D) 表示在第一个单词基础上出现第二个单词的概率，即条件概率，以此类推，最后是出现**< EOS >**的条件概率。

单个元素的softmax loss function为：

![L^{<t>}(\hat y^{<t>},y^{<t>})=-\sum_iy_i^{<t>}log\ \hat y_i^{<t>}](https://www.zhihu.com/equation?tex=L%5E%7B%3Ct%3E%7D%28%5Chat+y%5E%7B%3Ct%3E%7D%2Cy%5E%7B%3Ct%3E%7D%29%3D-%5Csum_iy_i%5E%7B%3Ct%3E%7Dlog%5C+%5Chat+y_i%5E%7B%3Ct%3E%7D)

该样本所有元素的Loss function为：

![L(\hat y,y)=\sum_tL^{<t>}(\hat y^{<t>},y^{<t>})](https://www.zhihu.com/equation?tex=L%28%5Chat+y%2Cy%29%3D%5Csum_tL%5E%7B%3Ct%3E%7D%28%5Chat+y%5E%7B%3Ct%3E%7D%2Cy%5E%7B%3Ct%3E%7D%29)

对语料库的每条语句进行RNN模型训练，最终得到的模型可以根据给出语句的前几个单词预测其余部分，将语句补充完整。例如给出**“Cats average 15”**，RNN模型可能预测完整的语句是**“Cats average 15 hours of sleep a day.”**。

最后补充一点，整个语句出现的概率等于语句中所有元素出现的条件概率乘积。例如某个语句包含 ![y^{<1>},y^{<2>},y^{<3>}](https://www.zhihu.com/equation?tex=y%5E%7B%3C1%3E%7D%2Cy%5E%7B%3C2%3E%7D%2Cy%5E%7B%3C3%3E%7D) ，则整个语句出现的概率为：

![P(y^{<1>},y^{<2>},y^{<3>})=P(y^{<1>})\cdot P(y^{<2>}|y^{<1>})\cdot P(y^{<3>}|y^{<1>},y^{<2>})](https://www.zhihu.com/equation?tex=P%28y%5E%7B%3C1%3E%7D%2Cy%5E%7B%3C2%3E%7D%2Cy%5E%7B%3C3%3E%7D%29%3DP%28y%5E%7B%3C1%3E%7D%29%5Ccdot+P%28y%5E%7B%3C2%3E%7D%7Cy%5E%7B%3C1%3E%7D%29%5Ccdot+P%28y%5E%7B%3C3%3E%7D%7Cy%5E%7B%3C1%3E%7D%2Cy%5E%7B%3C2%3E%7D%29)

## 7 Sampling novel sequences

利用训练好的RNN语言模型，可以进行新的序列采样，从而随机产生新的语句。与上一节介绍的一样，相应的RNN模型如下所示：

![](https://pic4.zhimg.com/80/v2-f2f5643b1c850e900d96c8f6dc5fe36f_hd.jpg)

首先，从第一个元素输出 ![\hat y^{<1>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3C1%3E%7D) 的softmax分布中随机选取一个word作为新语句的首单词。然后， ![y^{<1>}](https://www.zhihu.com/equation?tex=y%5E%7B%3C1%3E%7D) 作为 ![x^{<2>}](https://www.zhihu.com/equation?tex=x%5E%7B%3C2%3E%7D) ，得到 ![\hat y^{<1>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3C1%3E%7D) 的softmax分布。从中选取概率最大的word作为 ![y^{<2>}](https://www.zhihu.com/equation?tex=y%5E%7B%3C2%3E%7D) ，继续将 ![y^{<2>}](https://www.zhihu.com/equation?tex=y%5E%7B%3C2%3E%7D) 作为 ![x^{<3>}](https://www.zhihu.com/equation?tex=x%5E%7B%3C3%3E%7D) ，以此类推。直到产生**< EOS >**结束符，则标志语句生成完毕。当然，也可以设定语句长度上限，达到长度上限即停止生成新的单词。最终，根据随机选择的首单词，RNN模型会生成一条新的语句。

值得一提的是，如果不希望新的语句中包含**< UNK >**标志符，可以在每次产生**< UNK >**时重新采样，直到生成非**< UNK >**标志符为止。

以上介绍的是word level RNN，即每次生成单个word，语句由多个words构成。另外一种情况是character level RNN，即词汇表由单个英文字母或字符组成，如下所示：

![Vocabulay=[a,b,c,\cdots,z,.,;,\ ,0,1,\cdots,9,A,B,\cdots,Z]](https://www.zhihu.com/equation?tex=Vocabulay%3D%5Ba%2Cb%2Cc%2C%5Ccdots%2Cz%2C.%2C%3B%2C%5C+%2C0%2C1%2C%5Ccdots%2C9%2CA%2CB%2C%5Ccdots%2CZ%5D)

Character level RNN与word level RNN不同的是， ![\hat y^{<t>}](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D) 由单个字符组成而不是word。训练集中的每句话都当成是由许多字符组成的。character level RNN的优点是能有效避免遇到词汇表中不存在的单词**< UNK >**。但是，character level RNN的缺点也很突出。由于是字符表征，每句话的字符数量很大，这种大的跨度不利于寻找语句前部分和后部分之间的依赖性。另外，character level RNN的在训练时的计算量也是庞大的。基于这些缺点，目前character level RNN的应用并不广泛，但是在特定应用下仍然有发展的趋势。

## 8\. Vanisging gradients with RNNs

语句中可能存在跨度很大的依赖关系，即某个word可能与它距离较远的某个word具有强依赖关系。例如下面这两条语句：

**The** **cat, which already ate fish,** **was** **full.**

**The** **cats, which already ate fish,** **were** **full.**

第一句话中，was受cat影响；第二句话中，were受cats影响。它们之间都跨越了很多单词。而一般的RNN模型每个元素受其周围附近的影响较大，难以建立跨度较大的依赖性。上面两句话的这种依赖关系，由于跨度很大，普通的RNN网络容易出现梯度消失，捕捉不到它们之间的依赖，造成语法错误。关于梯度消失的原理，我们在之前的[Coursera吴恩达《优化深度神经网络》课程笔记（1）-- 深度学习的实用层面](https://zhuanlan.zhihu.com/p/30341532)已经有过介绍，可参考。

另一方面，RNN也可能出现梯度爆炸的问题，即gradient过大。常用的解决办法是设定一个阈值，一旦梯度最大值达到这个阈值，就对整个梯度向量进行尺度缩小。这种做法被称为gradient clipping。

## 9\. Gated Recurrent Unit(GRU)

RNN的隐藏层单元结构如下图所示：

![](https://pic1.zhimg.com/80/v2-8bc06dcf425362f0c6747c0096875ba2_hd.jpg)

![a^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D) 的表达式为：

![a^{<t>}=tanh(W_a[a^{<t-1>},x^{<t>}]+b_a)](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3Dtanh%28W_a%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_a%29)

为了解决梯度消失问题，对上述单元进行修改，添加了记忆单元，构建GRU，如下图所示：

![](https://pic2.zhimg.com/80/v2-c6d124138043c92c0d278145d9195a90_hd.jpg)

相应的表达式为：

![\tilde c^{<t>}=tanh(W_c[c^{<t-1>},x^{<t>}]+b_c)](https://www.zhihu.com/equation?tex=%5Ctilde+c%5E%7B%3Ct%3E%7D%3Dtanh%28W_c%5Bc%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_c%29)

![\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)](https://www.zhihu.com/equation?tex=%5CGamma_u%3D%5Csigma%28W_u%5Bc%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_u%29)

![c^{<t>}=\Gamma*\tilde c^{<t>}+(1-\Gamma_u)*c^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3D%5CGamma%2A%5Ctilde+c%5E%7B%3Ct%3E%7D%2B%281-%5CGamma_u%29%2Ac%5E%7B%3Ct-1%3E%7D)

其中， ![c^{<t-1>}=a^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct-1%3E%7D%3Da%5E%7B%3Ct-1%3E%7D) ， ![c^{<t>}=a^{<t>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3Da%5E%7B%3Ct%3E%7D) 。 ![\Gamma_u](https://www.zhihu.com/equation?tex=%5CGamma_u) 意为gate，记忆单元。当 ![\Gamma_u=1](https://www.zhihu.com/equation?tex=%5CGamma_u%3D1) 时，代表更新；当 ![\Gamma_u=0](https://www.zhihu.com/equation?tex=%5CGamma_u%3D0) 时，代表记忆，保留之前的模块输出。这一点跟CNN中的ResNets的作用有点类似。因此， ![\Gamma_u](https://www.zhihu.com/equation?tex=%5CGamma_u) 能够保证RNN模型中跨度很大的依赖关系不受影响，消除梯度消失问题。

上面介绍的是简化的GRU模型，完整的GRU添加了另外一个gate，即 ![\Gamma_r](https://www.zhihu.com/equation?tex=%5CGamma_r) ，表达式如下：

![\tilde c^{<t>}=tanh(W_c[\Gamma_r*c^{<t-1>},x^{<t>}]+b_c)](https://www.zhihu.com/equation?tex=%5Ctilde+c%5E%7B%3Ct%3E%7D%3Dtanh%28W_c%5B%5CGamma_r%2Ac%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_c%29)

![\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)](https://www.zhihu.com/equation?tex=%5CGamma_u%3D%5Csigma%28W_u%5Bc%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_u%29)

![\Gamma_r=\sigma(W_r[c^{<t-1>},x^{<t>}]+b_r)](https://www.zhihu.com/equation?tex=%5CGamma_r%3D%5Csigma%28W_r%5Bc%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_r%29)

![c^{<t>}=\Gamma*\tilde c^{<t>}+(1-\Gamma_u)*c^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3D%5CGamma%2A%5Ctilde+c%5E%7B%3Ct%3E%7D%2B%281-%5CGamma_u%29%2Ac%5E%7B%3Ct-1%3E%7D)

![a^{<t>}=c^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3Dc%5E%7B%3Ct%3E%7D)

注意，以上表达式中的 ![*](https://www.zhihu.com/equation?tex=%2A) 表示元素相乘，而非矩阵相乘。

## 10\. Long Short Term Memory(LSTM)

LSTM是另一种更强大的解决梯度消失问题的方法。它对应的RNN隐藏层单元结构如下图所示：

![](https://pic4.zhimg.com/80/v2-d98c23a792367ddd65d6e02995c74e7d_hd.jpg)

相应的表达式为：

![\tilde c^{<t>}=tanh(W_c[a^{<t-1>},x^{<t>}]+b_c)](https://www.zhihu.com/equation?tex=%5Ctilde+c%5E%7B%3Ct%3E%7D%3Dtanh%28W_c%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_c%29)

![\Gamma_u=\sigma(W_u[a^{<t-1>},x^{<t>}]+b_u)](https://www.zhihu.com/equation?tex=%5CGamma_u%3D%5Csigma%28W_u%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_u%29)

![\Gamma_f=\sigma(W_f[a^{<t-1>},x^{<t>}]+b_f)](https://www.zhihu.com/equation?tex=%5CGamma_f%3D%5Csigma%28W_f%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_f%29)

![\Gamma_o=\sigma(W_o[a^{<t-1>},x^{<t>}]+b_o)](https://www.zhihu.com/equation?tex=%5CGamma_o%3D%5Csigma%28W_o%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_o%29)

![c^{<t>}=\Gamma_u*\tilde c^{<t>}+\Gamma_f*c^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3D%5CGamma_u%2A%5Ctilde+c%5E%7B%3Ct%3E%7D%2B%5CGamma_f%2Ac%5E%7B%3Ct-1%3E%7D)

![a^{<t>}=\Gamma_o*c^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3D%5CGamma_o%2Ac%5E%7B%3Ct%3E%7D)

LSTM包含三个gates： ![\Gamma_u](https://www.zhihu.com/equation?tex=%5CGamma_u) ， ![\Gamma_f](https://www.zhihu.com/equation?tex=%5CGamma_f) ， ![\Gamma_o](https://www.zhihu.com/equation?tex=%5CGamma_o) ，分别对应update gate，forget gate和output gate。

如果考虑 ![c^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct-1%3E%7D) 对 ![\Gamma_u](https://www.zhihu.com/equation?tex=%5CGamma_u) ， ![\Gamma_f](https://www.zhihu.com/equation?tex=%5CGamma_f) ， ![\Gamma_o](https://www.zhihu.com/equation?tex=%5CGamma_o) 的影响，可加入peephole connection，对LSTM的表达式进行修改：

![\tilde c^{<t>}=tanh(W_c[a^{<t-1>},x^{<t>}]+b_c)](https://www.zhihu.com/equation?tex=%5Ctilde+c%5E%7B%3Ct%3E%7D%3Dtanh%28W_c%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_c%29)

![\Gamma_u=\sigma(W_u[a^{<t-1>},x^{<t>},c^{<t-1>}]+b_u)](https://www.zhihu.com/equation?tex=%5CGamma_u%3D%5Csigma%28W_u%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%2Cc%5E%7B%3Ct-1%3E%7D%5D%2Bb_u%29)

![\Gamma_f=\sigma(W_f[a^{<t-1>},x^{<t>},c^{<t-1>}]+b_f)](https://www.zhihu.com/equation?tex=%5CGamma_f%3D%5Csigma%28W_f%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%2Cc%5E%7B%3Ct-1%3E%7D%5D%2Bb_f%29)

![\Gamma_o=\sigma(W_o[a^{<t-1>},x^{<t>},c^{<t-1>}]+b_o)](https://www.zhihu.com/equation?tex=%5CGamma_o%3D%5Csigma%28W_o%5Ba%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%2Cc%5E%7B%3Ct-1%3E%7D%5D%2Bb_o%29)

![c^{<t>}=\Gamma_u*\tilde c^{<t>}+\Gamma_f*c^{<t-1>}](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3D%5CGamma_u%2A%5Ctilde+c%5E%7B%3Ct%3E%7D%2B%5CGamma_f%2Ac%5E%7B%3Ct-1%3E%7D)

![a^{<t>}=\Gamma_o*c^{<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%3Ct%3E%7D%3D%5CGamma_o%2Ac%5E%7B%3Ct%3E%7D)

GRU可以看成是简化的LSTM，两种方法都具有各自的优势。

## 11\. Bidirectional RNN

我们在第3节中简单提过Bidirectional RNN，它的结构如下图所示：

![](https://pic2.zhimg.com/80/v2-684aef09b288d34f5f77ad121c0f1157_hd.jpg)

BRNN对应的输出 ![y^{<t>}](https://www.zhihu.com/equation?tex=y%5E%7B%3Ct%3E%7D) 表达式为：

![\hat y^{<t>}=g(W_{y}[a^{\rightarrow <t>},a^{\leftarrow <t>}]+b_y)](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%3Ct%3E%7D%3Dg%28W_%7By%7D%5Ba%5E%7B%5Crightarrow+%3Ct%3E%7D%2Ca%5E%7B%5Cleftarrow+%3Ct%3E%7D%5D%2Bb_y%29)

BRNN能够同时对序列进行双向处理，性能大大提高。但是计算量较大，且在处理实时语音时，需要等到完整的一句话结束时才能进行分析。

## 12\. Deep RNNs

Deep RNNs由多层RNN组成，其结构如下图所示：

![](https://pic3.zhimg.com/80/v2-634a826cfbfe7c1e021528c0ff5ddf9b_hd.jpg)

与DNN一样，用上标 ![[l]](https://www.zhihu.com/equation?tex=%5Bl%5D) 表示层数。Deep RNNs中 ![a^{[l]<t>}](https://www.zhihu.com/equation?tex=a%5E%7B%5Bl%5D%3Ct%3E%7D) 的表达式为：

![a^{[l]<t>}=g(W_a^{[l]}[a^{[l]<t-1>},a^{[l-1]<t>}]+b_a^{[l]})](https://www.zhihu.com/equation?tex=a%5E%7B%5Bl%5D%3Ct%3E%7D%3Dg%28W_a%5E%7B%5Bl%5D%7D%5Ba%5E%7B%5Bl%5D%3Ct-1%3E%7D%2Ca%5E%7B%5Bl-1%5D%3Ct%3E%7D%5D%2Bb_a%5E%7B%5Bl%5D%7D%29)

我们知道DNN层数可达100多，而Deep RNNs一般没有那么多层，3层RNNs已经较复杂了。

另外一种Deep RNNs结构是每个输出层上还有一些垂直单元，如下图所示：

![](https://pic3.zhimg.com/80/v2-685d20dbc0f6455931a622d4b787eda5_hd.jpg)

至此，第一节笔记介绍完毕！

**更多AI资源请关注公众号：红色石头的机器学习之路（ID：redstonewill）**