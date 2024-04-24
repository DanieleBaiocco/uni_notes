## Introduction
Viral marketing exploits existing social networks by encouraging customers to share product information with their friends.
Until recently it has been difficult to measure how influential person-to-person recommendations actually are over a wide range of products.
We were able to directly measure and model the effectiveness of recommendations by studying one online retailer’s incentivised viral marketing program.
The website gave discounts to customers recommending any of its products to others, and then tracked the resulting purchases and additional recommendations.
We find (WE WILL SHOW IT) that product purchases that result from recommendations are not far from the usual 80-20 rule. The rule states that the top twenty percent of the products account for 80 percent of the sales. In our case the top 20% of the products contribute to about half the sales.
It is human nature to be more interested in what a friend buys than what an anonymous person buys, to be more likely to trust their opinion, and to be more influenced by their actions.
As one would expect our friends are also acquainted with our needs and tastes, and can make appropriate recommendations.
A Lucid Marketing survey found that 68% of individuals consulted friends and relatives before purchasing home electronics 
In our study we are able to directly  **observe** the effectiveness of person to person **word of mouth** advertising for hundreds of thousands of products for the first time.

We find that most recommendation chains do not grow very large, often terminating with the initial purchase of a product. However, occasionally a product will propagate through a very active recommendation network. We propose a simple stochastic model that seems to explain the propagation of recommendations.

Moreover, the characteristics of recommendation networks influence the purchase patterns of their members. For example, individuals’ likelihood of purchasing a product initially increases as they receive additional recommendations for it, but a saturation point is quickly reached.

Interestingly, as more recommendations are sent between the same two individuals, the likelihood that they will be heeded decreases.

We find that communities (automatically found by graph theoretic community finding algorithm) were usually centered around a product group, such as books, music, or DVDs, but almost all of them shared recommendations for all types of products. 

We also find patterns of homophily, the tendency of like to associate with like, with communities of customers recommending types of products reflecting their common interests

We propose models to identify products for which viral marketing is effective: We find that the category and price of product plays a role, with recommendations of expensive products of interest to small, well connected communities resulting in a purchase more often

We also observe patterns in the timing of recommendations and purchases corresponding to times of day when people are likely to be shopping online or reading email.

## Related Work
Viral marketing can be thought of as a diffusion of information about the product and its adoption over the network.

The classical disease propagation models are based on the stages of a disease in a host: a person is first susceptible to a disease, then if she is exposed to an infectious contact she can become infected and thus infectious. After the disease ceases the person is recovered or removed. Person is then immune for some period. The immunity can also wear off and the person becomes again susceptible. Thus SIR (susceptible – infected – recovered) models diseases where a recovered person never again becomes susceptible, while SIRS (SIS, susceptible – infected – (recovered) – susceptible) models population in which recovered host can become susceptible again.

Given a network and a set of infected nodes the epidemic threshold is studied, i.e. conditions under which the disease will either dominate or die out. In our case SIR model would correspond to the case where a set of initially infected nodes corresponds to people that purchased a product without first receiving the recommendations.

A node can purchase a product only once, and then tries to infect its neighbors with a purchase by sending out the recommendations. SIS model corresponds to less realistic case where a person can purchase a product multiple times as a result of multiple recommendations.

The problem with these type of models is that they assume a known social network over which the diseases (product recommendations) are spreading and usually a single parameter which specifies the infectiousness of the disease. In our context this would mean that the whole population is equally susceptible to recommendations of a particular product.

There are numerous other models of influence spread in social networks. One of the first and most influential diffusion models was proposed by Bass. The model of product diffusion predicts the number of people who will adopt an innovation over time. It does not explicitly account for the structure of the social network but it rather assumes that the rate of adoption is a function of the current proportion of the population who have already adopted (purchased a product in our case). The diffusion equation models the cumulative proportion of adopters in the population as a function of the intrinsic adoption rate, and a measure of social contagion. The model describes an S-shaped curve, where adoption is slow at first, takes off exponentially and flattens at the end. It can effectively model word-of-mouth product diffusion at the aggregate level, but not at the level of an individual person, which is one of the topics we explore in this paper.

Diffusion models that try to model the process of adoption of an idea or a product can generally be divided into two groups:
1. **Threshold model** where each node in the network has a threshold t ∈ [0, 1], typically drawn from some probability distribution. We also assign connection weights wu,v on the edges of the network. A node adopts the behavior if a sum of the connection weights of its neighbors that already adopted the behavior (purchased a product in our case) is greater than the threshold t.
2. **Cascade model** where whenever a neighbor v of node u adopts, then node u also adopts with probability pu,v. In other words, every time a neighbor of u purchases a product, there is a chance that u will decide to purchase as well.
Compared to previous empirical studies which tracked the adoption of a single innovation or product, our data encompasses over half a million different products, allowing us to model a product’s suitability for viral marketing in terms of both the properties of the network and the product itself.
## The Recommendation Network
Our analysis focuses on the recommendation referral program run by a large retailer. The program rules were as follows:
1. . Each time a person purchases a book, music, or a movie he or she is given the option of sending emails recommending the item to friends.
2. The first person to purchase the same item through a referral link in the email gets a 10% discount.
3. When this happens the sender of the recommendation receives a 10% credit on their purchase.
The following information is recorded for each recommendation:
1. Sender Customer ID (shadowed)
2. Receiver Customer ID (shadowed)
3. Date of Sending
4. Purchase flag (buy-bit) (it tells whether the reciever WAS THE FIRST ONE TO purchased the product or not)
6. Purchase Date
7. Product identifier
8. Price
The recommendation dataset consists of 15,646,121 recommendations made among 3,943,084 distinct users
The data was collected from June 5 2001 to May 16 2003.
In total, 548,523 products were recommended, 99% of them belonging to 4 main product groups: Books, DVDs, Music and Videos.
In addition to recommendation data, we also crawled the retailer’s website to obtain product categories, reviews and ratings for all products. Of the products in our data set, 5813 (1%) were discontinued (the retailer no longer provided any information about them).

Although the data gives us a detailed and accurate view of recommendation dynamics, it does have its limitations. **The only indication of the success of a recommendation is the observation of the recipient purchasing the product through the same vendor.**
We have no way of knowing if the person had decided instead to purchase elsewhere, borrow, or otherwise obtain the product.

The delivery of the recommendation is also somewhat different from one person simply telling another about a product they enjoy, possibly in the context of a broader discussion of similar products

The recommendation is received as a form email including information about the discount program. Someone reading the email might consider it spam, or at least deem it less important than a recommendation given in the context of a **conversation**.


The recipient may also doubt whether the friend is recommending the product because they think the recipient might enjoy it, or are simply trying to get a discount for themselves.

Finally, because the recommendation takes place before the recommender receives the product, it might not be based on a direct observation of the product. Nevertheless, we believe that these recommendation networks are reflective of the nature of word of mouth advertising

### Identifying successfull recomendations
For each recommendation, the dataset includes information about the recommended product, sender and received or the recommendation, and most importantly, the **success of recommendation.**

We represent this data set as a directed multi graph.
The nodes represent customers, and a directed edge contains all the information about the recommendation. The edge (i, j, p, t) indicates that i recommended product p to customer j at time t.

Note that as there can be multiple recommendations of between the persons (even on the same product) there can be **multiple edges between two nodes**.

The typical process generating edges in the recommendation network is as follows: a node i first buys a product p at time t and then it recommends it to nodes j1, . . . , jn. The j nodes can then buy the product and further recommend it. **The only way for a node to recommend a product is to first buy it.**

Note that even if all nodes j buy a product, only the edge to the node jk that first made the purchase (within a week after the recommendation) will be marked by a buy-bit.

NOTA Because the buy-bit is set only for the first person who acts on a recommendation, we identify additional purchases by the presence of outgoing recommendations for a person, since all recommendations must be preceded by a purchase. We call this type of evidence of purchase a buy-edge.

NOTA It is possible for a customer to not be the first to act on a recommendation and also to not recommend the product to others ( significa che un reciever puo tranquillamente comprare il prodotto NON PER PRIMO e NON generare nessuna forma di raccomandazione ).
Cio non e' stato registrato nel DS sfortunatamente.
We consider, however, the buy-bits and buy-edges as proxies for the total number of purchases through recommendations.


To avoid confusion we will refer to edges in a multi graph as recommendations (or multi-edges) — there can be more than one recommendation between a pair of nodes. We will use the term edge (or unique edge) to refer to edges in the usual sense, i.e. there is only one edge between a pair of people. And, to get from recommendations to edges we create an edge between a pair of people if they exchanged **at least one recommendation**.

FIRST EXPERIMENT
Table 1 shows the sizes of various product group recommendation networks with p being the total number of products in the product group, n the total number of nodes spanned by the group recommendation network, and r the number of recommendations (there can be multiple recommendations between two nodes). Column e shows the number of (unique) edges – disregarding multiple recommendations between the same source and recipient (i.e., number of pairs of people that exchanged at least one recommendation).

In terms of the number of different items, there are by far the most music CDs, followed by books and videos. There is a surprisingly small number of DVD titles. On the other hand, DVDs account for more half of all recommendations in the dataset.
![[network_statistics_1.png]]
Music recommendations reached about the same number of people as DVDs but used more than 5 times fewer recommendations to achieve the same coverage of the nodes. Book recommendations reached by far the most people (2.8 milioni). Notice that all networks have a very small number of unique edges. 

IMPORTANTE
For books, videos and music the number of unique edges is smaller than the number of nodes – this suggests that the networks are highly disconnected.

IMPORTANTE
given the total number of recommendations r and purchases (bb + be) influenced by recommendations we can estimate how many recommendations need to be independently sent over the network to induce a new purchase. Using this metric books have the most influential recommendations followed by DVDs and music. For books one out of 69 recommendations resulted in a purchase. For DVDs it increases to 108 recommendations per purchase and further increases to 136 for music and 203 for video

IF I TAKE INTO CONSIDERATION THE STUCTURE OF THE LARGEST COMPONENT of each product group’s recommendation network I have this 
![[network_statistics_2.png]]
First, notice that the largest connected components are very small.
DVDs have the largest - containing 4.9% of the nodes, books have the smallest at 1.78%.

One would also expect that the fraction of the recommendations in the largest component would be proportional to its size. We notice that this is not the case.
for DVDs 84.3% of the recommendations are in largest component (which contains 4.9% of all DVD nodes) vs. 16.3% for book recommendations (component size 1.79%), 20.5% for music recommendations (component size 2.77%), and 8.4% for video recommendations (component size 2.1%).
This shows that the dynamic in the largest component is very much different from the rest of the network. Especially **for DVDs we can see that a very small fraction of users generated most of the recommendations.**

### Recommendation network over time
Voglio vedere se sta network rispecchia una SMALL WORLD NETWORK.
We examined whether the edges formed by aggregating recommendations over all products would similarly yield a small world network, even though they represent only a small fraction of a person’s complete social network.
Within the weakly connected component, any node can be reached from any other node by traversing (undirected) edges. For example, if u recommended product x to v, and w recommended product y to v, then uand w are linked through one intermediary and thus belong to the same weakly connected component.

Nodes belong to same component if they can reach each other via an undirected path regardless of how densely they are linked.

We measured the growth of the largest weakly connected component over time, shown in Figure 1.
Figure 1 shows the size of the largest connected component, as a fraction of the total network. The largest component is very small over all time. Even though we compose the network using all the recommendations in the dataset, the largest connected component contains less than 2.5% (100,420) of the nodes, and the second largest component has only 600 nodes.

The insert in figure 1 shows the growth of the customer base over time. Surprisingly it was linear, adding on average 165,000 new users each month, which is an indication that the service itself was not spreading epidemically
![[network_statistics_3.png]]
#### Growth of the largest connected component
Here we are not interested in how fast the largest component grows over time but rather how big other components are when they get merged into the largest component.

Also, since our graph is directed we are interested in determining whether smaller components become attached to the largest component by a recommendation sent from inside of the largest component.

The other possibility is that the recommendation comes from a node outside the component to a member of the largest component and thus the initiative to attach comes from outside the largest component.

We look at whether the largest component grows gradually, adding nodes one by one as the members send out more recommendations, or whether a new recommendation might act as a bridge to a component consisting of several nodes who are already linked by their previous recommendations 
 IMPORTANTE CAPIRE CHE STO FACENDO QUESTO: To this end we measure the distribution of a component’s size when it gets merged to the largest weakly connected component.
 Recommendations are arriving over time one by one creating edges between the nodes of the network. As more edges are being added the size of largest connected component grows. We keep track of the currently largest component, and measure how big the separate components are when they get attached to the largest component.
 
Figure 2(a) shows the distribution of merged connected component (CC) sizes. On the x-axis we plot the component size (number of nodes N) and on the y-axis the number of components of size N that were merged over time with the largest component. We see that a majority of the time a single node (component of size 1) merged with the currently largest component.
![[network_statistic_4.png]]
We see that a majority of the time a single node (component of size 1) merged with the currently largest component. On the other extreme is the case when a component of 1, 568 nodes merged with the largest component.

Interestingly, out of all merged components, in 77% of the cases the source of the recommendation comes from inside the largest component, while in the remaining 23% of the cases it is the smaller component that attaches itself to the largest one.

Figure 2(b) shows the distribution of component sizes only for the case when the sender of the recommendation was a member of the largest component, i.e. the small component was attached from the largest component. 
![[network_statistic_5.png]]

Lastly, Figure 2(c) shows the distribution for the opposite case when the sender of the recommendation was not a member of the largest component, i.e. the small component attached itself to the largest
![[network_statistic_6.png]]


Also notice that in all cases the distribution of merged component sizes follows a heavy-tailed distribution

RISULTATI: We fit a power-law distribution and note the power-law exponent of 1.90 (fig. 2(a)) when considering all merged components. Limiting the analysis to the cases where the source of the edge that attached a small component to the largest is in the largest component we obtain power-law exponent of 1.96 (fig. 2(b)), and when the edge originated from the small component to attached it to the largest, the power-law exponent is 1.76.

This shows that even though in most cases the LCC absorbs the small component, we see that components that attach themselves to the LCC tend to be larger (smaller power-law exponent) than those attracted by the LCC.

**This means that the component sometimes grows a bit before it attaches itself to the largest component.**

Intuitively, an individual node can get attached to the largest component simply by passively receiving a recommendation. But if it is the outside node that sends a recommendation to someone in the giant component, it is already an active recommender and could therefore have recommended to several others previously, thus forming a slightly bigger component that is then merged.

From these experiments we see that the largest component is very active, adding smaller components by generating new recommendations. Most of the time these newly merged components are quite small, but occasionally sizable components are attached.

## Observations from the previous statistical results
**It seems that some people got quite heavily involved in the recommendation program, and that they tended to recommend a large number of products to the same set of friends (since the number of unique edges is so small as shown on table 1).**
This means that people tend to buy more DVDs and also like to recommend them to their friends, while they seem to be more conservative with books.

## Propagation of recommendations
Not all people who accept a recommendation by making a purchase also decide to give recommendations.
Table 3 shows that only about a third of the people that purchase also recommend the product forward.
The ratio of forward recommendations is much higher for DVDs than for other kinds of products. Videos also have a higher ratio of forward recommendations, while books have the lowest. This shows that people are most keen on recommending movies, possibly for the above mentioned reasons, while more conservative when recommending books and music.
![[network_statistic_7.png]]
Figure 4 shows the cumulative out-degree distribution, that is the number of people who sent out at least kp recommendations, for a product. Also, notice the exponential decay in the tail of the distribution which could be, among other reasons, attributed to the finite time horizon of our dataset.
![[network_statistic_8.png]]
The figure 4 shows that the deeper an individual is in the cascade, if they choose to make recommendations, **they tend to recommend to a greater number of people on average ** (the fitted line has smaller slope γ, i.e. the distribution has higher variance).


We also observe, as is shown in Table 4, that the probability of an individual making a recommendation at all (which can only occur if they make a purchase), declines after an initial increase as one gets deeper into the cascade.

![[network_statistic_9.png]]
## Identifying cascades
As customers continue forwarding recommendations, they contribute to the formation of cascades. In order to identify cascades, i.e. the “causal” propagation of recommendations, we track successful recommendations as they influence purchases and further recommendations.
We define a recommendation to be successful if it reached a node before its first purchase. We consider only the first purchase of an item, because there are many cases when a person made multiple purchases of the same product, and in between those purchases she may have received new recommendations.

Each cascade is a network consisting of customers (nodes) who purchased the same product as a result of each other’s recommendations (edges).

We delete late recommendations — all incoming recommendations that happened after the first purchase of the product. This way we make the network time increasing or causal — for each node all incoming edges (recommendations) occurred before all outgoing edges.
Now each connected component represents a time obeying propagation of recommendations.

Most product recommendation networks consist of a large number of small disconnected components where **we do not observe cascades**.
Here is an example:
![[medical_book_cascade.png]]

Then there is usually a small number of relatively small components with recommendations successfully propagating, like for this graphic novel where Some nodes recommend to many friends, forming a star like pattern.:
![[graphic_novel_cascade.png]]

*The distribution of these cascade networks  is reflected in the heavy tailed distribution of cascade sizes , having a power-law exponent close to 1 for DVDs in particular. We determined the power-law exponent by fitting a line on log-log scales using the least squares method*.
![[network_statistics_10.png]]

The following figure shows the distribution of the recommendations and purchases made by a single node in the recommendation network.
![[network_statistic_11.png]]
Notice the power-law distributions and long flat tails. The most active customer made 83,729 recommendations and purchased 4,416 different items.

Last, we examine the number of exchanged recommendations between a pair of people in figure 7 (e' diverso da sopra perche qui vedo quante raccomandazioni di media si scambiano tra due persone, quidni magari due persone si scambiano in media 4 raccomandazioni ma ovviamnete l'outdegree delle raccomandazioni non e' legato a sta cosa). 
![[network_statistic_12.png]]
Overall, 39% of pairs of people exchanged just a single recommendation. This number decreases for DVDs to 37%, and increases for books to 45%. The distribution of the number of exchanged recommendations follows a heavy tailed distribution. To get a better understanding of the distributions we show the power-law decay lines. Notice that one gets much stronger decay exponent (distribution has weaker tail) of -2.7 for books and a very shallow power-law exponent of -1.5 for DVDs. This means that even a pair of people exchanges more DVD than book recommendations

## The recomendation propagation model
A simple model can help explain how the wide variance we observe in the number of recommendations made by individuals can lead to power-laws in cascade sizes (figure 6): ![[network_statistics_10.png]]
The model assumes that each recipient of a recommendation will forward it to others if its value exceeds an arbitrary threshold that the individual sets for herself. Since exceeding this value is a probabilistic event, let’s call p_t the probability that at time step t the recommendation exceeds the threshold. In that case the number of recommendations Nt+1 at time (t + 1) is given in terms of the number of recommendations at an earlier time by
N_t+1 = p_t x N_t

Subtracting from both sides of this equation the term Nt and diving by it we obtain
![[eq1.png]]
Summing both sides from the initial time to some very large time T and assuming that for long times the numerator is smaller than the denominator (a reasonable assumption) we get, up to a unit constant
![[eq2.png]]
The left hand integral is just log(N), and the right hand side is a sum of random variables, which in the limit of a very large uncorrelated number of recommendations is normally distributed (central limit theorem).
This means that the logarithm of the number of messages is normally distributed, or equivalently, that the number of messages passed is log-normally distributed. In other words the probability density for N is given by
![[eq3.png]]
Furthermore, for large variances, the lognormal distribution can behave like a power law for a range of values. In order to see this, take the logarithms on both sides of the equation (equivalent to a log-log plot) and one obtains
![[eq4.png]]
So, for large σ, the last term of the right hand side goes to zero, and since the second term is a constant one obtains a power law behavior with exponent value of minus one

## Success of recomendation
So far we only looked into the aggregate statistics of the recommendation network. Next, we ask questions about the effectiveness of recommendations in the recommendation network itself.
### 1 : Success of subsequent recomendation vs number of incoming recomendatiosn
First, we examine how the probability of purchasing changes as one gets more and more recommendations. One would expect that a person is more likely to buy a product if she gets more recommendations. On the other had one would also think that there is a saturation point – if a person hasn’t bought a product after a number of recommendations, they are not likely to change their minds after receiving even more of them. So, how many recommendations are too many?
Figure 8 shows the probability of purchasing a product as a function of the number of incoming recommendations on the product. 
![[figure_1.png]]
Because we exclude late recommendations, those that were received after the purchase, an individual counts as having received three recommendations only if they did not make a purchase after the first two, and either purchased or did not receive further recommendations after receiving the third one.

NOTA 
As we move to higher numbers of incoming recommendations, the number of observations drops rapidly. For example, there were 5 million cases with 1 incoming recommendation on a book, and only 58 cases where a person got 20 incoming recommendations on a particular book. For these reasons we cut-off the plot when the number of observations becomes too small and the error bars too large.

We calculate the purchase probabilities and the standard errors of the estimates which we use to plot the error bars in the following way. We regard each point as a binomial random variable. Given the number of observations n, let m be the number of successes, and k (k=n-m) the number of failures. In our case, m is the number of people that first purchased a product after receiving r recommendations on it, and k is the number of people that received the total of r recommendations on a product (till the end of the dataset) but did purchase it, then the estimated probability of purchasing is phat = m/n and  the standard error s_phat of estimate phat is s_phat = sqrt(p(1 − p)/n).

Figure 8(a) shows that, overall, book recommendations are rarely followed. Even more surprisingly, as more and more recommendations are received, their success decreases. We observe a peak in probability of buying at 2 incoming recommendations and then a slow drop. This implies that if a person doesn’t buy a book after the first recommendation, but receives another, they are more likely to be persuaded by the second recommendation. But thereafter, they are less likely to respond to additional recommendations, possibly because they perceive them as spam, are less susceptible to others’ opinions, have a strong opinion on the particular product, or have a different means of accessing it.

For DVDs (figure 8(b)) we observe a saturation around 10 incoming recommendations. This means that with each additional recommendation, a person is more and more likely to be persuaded - up to a point. After a person gets 10 recommendations on a particular DVD, their probability of buying does not increase anymore.
### 2: Success of subsequent recommendations (from the same person)
Next, we analyze how the effectiveness of recommendations changes as one received more and more recommendations from the same person. A large number of exchanged recommendations can be a sign of trust and influence, but a sender of too many recommendations can be perceived as a spammer. A person who recommends only a few products will have her friends’ attention, but one who floods her friends with all sorts of recommendations will start to loose her influence.

We construct the experiment in the following way. For every recommendation r on some product p between nodes u and v, we first determine how many recommendations node v received from u before getting r. Then we check whether v, the recipient of recommendation, purchased p after the recommendation r arrived. If so, we count the recommendation as successful since it influenced the purchase.

This way we can calculate the recommendation success rate as more recommendations were exchanged. For the experiment we consider only node pairs (u, v), where there were at least a total of 10 recommendations sent from u to v . We perform the experiment using only recommendations from the same product group.
Figure 9 shows the probability of buying as a function of the total number of received recommendations from a particular person up to that point. One can think of x-axis as measuring time where the unit is the number of received recommendations from a particular person.
![[network_statistic_13.png]]
For books we observe that the effectiveness of recommendation remains about constant up to 3 exchanged recommendations. As the number of exchanged recommendations increases, the probability of buying starts to decrease to about half of the original value and then levels off. For DVDs we observe an immediate and consistent drop.
This experiment shows that recommendations start to lose effect after more than two or three are passed between two people. Also, notice that the effectiveness of book recommendations decays much more slowly than that of DVD recommendations, flattening out at around 20 recommendations, compared to around 10 DVD exchanged recommendations.

The result has important implications for viral marketing because providing too much incentive for people to recommend to one another can weaken the very social network links that the marketer is intending to exploit.

### 3: Success of outgoing recommendations (dal punto di vista del sender)
In previous sections we examined the data from the viewpoint of the receiver of the recommendation. Now we look from the viewpoint of the sender. The two interesting questions are:
1. how does the probability of getting a 10% credit change with the number of outgoing recommendations
2. given a number of outgoing recommendations, how many purchases will they influence?
One would expect that recommendations would be the most effective when recommended to the right subset of friends. If one is very selective and recommends to too few friends, then the chances of success are slim. One the other hand, recommending to everyone and spamming them with recommendations may have limited returns as well.

![[network_statistic_14.png]]The top row of figure 10 shows how the average number of purchases changes with the number of outgoing recommendations. For books, music, and videos the number of purchases soon saturates: it grows fast up to around 10 outgoing recommendations and then the trend either slows or starts to drop. DVDs exhibit different behavior, with the expected number of purchases increasing throughout. These results are even more interesting since the receiver of the recommendation does not know how many other people also received the recommendation. Thus the plots of figure 10 show that there are interesting dependencies between the product characteristics and the recommender that manifest through the number of recommendations sent.

Plotting the probability of getting a 10% credit as a function of the number of outgoing recommendations, as in the bottom row of figure 10, we see that the success of DVD recommendations saturates as well, while books, videos and music have qualitatively similar trends. The difference in the curves for DVD recommendations points to the presence of collisions in the dense DVD network, which has 10 recommendations per node and around 400 per product — an order of magnitude more than other product groups. This means that many different individuals are recommending to the same person, and after that person makes a purchase, even though all of them made a ‘successful recommendation’ by our definition, only one of them receives a credit.

### Proba of buying given  a number of different products a node got recommendations on
Im interested now in how does the behavior of customers change as they get more involved into the recommendation network? We would expect that most of the people are not heavily involved, so their probability of buying is not high.

There are two ways to measure the involvedness of a person in the network: by the total number of incoming recommendations (on all products) or the total number of different products they were recommended (NOTA CHE PRIMA AVEVI FATTO il calcolo solo su raccomandazioni legate a un determinato tipo di prodotto, ORA e' su qualsiasi tipo di prodotto).  
Usero' il secondo approccio (contando il total number of different products users got recommended).
For every purchase of a book at time t, we count the number of different books (DVDs, ...) the person received recommendations for before time t. As in all previous experiments we delete late recommendations, i.e. recommendations that arrived after the first purchase of a product.

![[network_statistic_15.png]]

We observe two distinct trends. For books and music (figures 11 (a) and (c)) the probability of buying is the highest when a person got recommendations on just 1 item, as the number of incoming recommended products increases to 2 or more the probability of buying quickly decreases and then flattens. Movies (DVDs and videos) exhibit different behavior (figure 11  (b) and (d)). A person is more likely to buy the more recommendations she gets. For DVDs the peak is at around 15 incoming products, while for videos there is no such peak – the probability remains fairly level.

Interestingly for DVDs the distribution reaches its low at 2 and 3 items, while for videos it lies somewhere between 3 and 8 items. The results suggest that books and music buyers tend to be conservative and focused.

## Timing on recommendations and purchases
The number of purchases with discount is the high when the number of purchases is small.
![[figure_2.png]]
This means that most of discounted purchases happened in the morning when the traffic (number of purchases/recommendations) on the retailer’s website was low. This makes sense since most of the recommendations happened during the day, and if the person wanted to get the discount by being the first one to purchase, she had the highest chances when the traffic on the website was the lowest.
## Recommendations and communities of interest
n the following analysis, we use a community finding algorithm in order to discover the types of products that link customers and so define a community. The algorithm breaks up the component into parts, such that the modularity Q, Q = (number of edges within communities)−(expected number of such edges), is maximized.
In other words, the algorithm identifies communities such that individuals within those communities tend to preferentially exchange recommendations with one another.
Applying the algorithm to the largest component, we identify many small communities and a few larger ones. The largest contains 21,000 nodes, 5,000 of whom are senders of a relatively modest 335,000 recommendations.

Let pc be the proportion of all recommendations that fall within a particular product category c. Then for a set of individuals sending xg recommendations, we would expect by chance that xg ∗pc ± sqrt( xg ∗ pc ∗ (1 − pc) ) would fall within category c.
We note the product categories for which the observed number of recommendations in the community is **many standard deviations higher than expected**.

For example, compared to the background population, the largest community is focused on a wide variety of books and music. In contrast, the second largest community, involving 10,412 individuals (4,205 of whom are sending over 3 million recommendations), is predominantly focused on DVDs from many different genres, with no particular emphasis on anime. The anime community itself emerges as a highly unusual group of 1,874 users who exchanged over 3 million recommendations.
Perhaps the most interesting are the medium sized communities, some of which are listed in Table 5, having between 100 and 1000 members and often reflecting specific interests.
![[figure_3.png]]
One of communities with the most particular interests recommended not only business and investing books to one another, but also an unusual number of books on terrorism, bacteriology, and military history.

Going back to components in the network that were disconnected from the largest component, we find similar patterns of homophily, the tendency of like to associate with like. Two of the components recommended technical books about medicine, one focused on dance music, while some others predominantly purchased books on business and investing.Given more time, it is quite possible that one of the customers in one of these disconnected components would have received a recommendation from a customer within the largest component, and the two components would have merged. 
At the very least many communities, no matter their focus, will have recommendations for children’s books or movies, since children are a focus for a great many people
### Recommendation effectiveness by book category
on. First, we compare the relative number of recommendations to reviews posted on the site (column c_av/r_p1 of table 6). Surprisingly, we find that the number of people making personal recommendations was only a few times greater than the number of people posting a public review on the website. We observe that fiction books have relatively few recommendations compared to the number of reviews, while professional and technical books have more recommendations than reviews. This could reflect several factors. One is that people feel more confident reviewing fiction than technical books. Another is that they hesitate to recommend a work of fiction before reading it themselves, since the recommendation must be made at the point of purchase. Yet another explanation is that the median price of a work of fiction is lower than that of a technical book. This means that the discount received for successfully recommending a mystery novel or thriller is lower and hence people have less incentive to send recommendations.
![[figure_4.png]]
Next, we measure the per category efficacy of recommendations by observing the ratio of the number of purchases occurring within a week following a recommendation to the number of recommenders for each book subject category. We observe marked differences in the response to recommendation for different categories of books. Fiction in general is not very effectively recommended, with only around 2% of recommenders succeeding. The efficacy was a bit higher (around 3%) for non-fiction books dealing with personal and leisure pursuits. Perhaps people generally know what their friends’ leisure interests are, or even have gotten to know them through those shared interests. On the other hand they may not know as much about each others’ tastes in fiction. Recommendation success is highest in the professional and technical category. Medical books have nearly double the average rate of recommendation acceptance. This could be in part attributed to the higher median price of medical books and technical books in general. As we will see in Section 9.2, a higher product price increases the chance that a recommendation will be accepted.

Recommendations are also more likely to be accepted for certain religious categories: 4.3% for Christian living and theology and 4.8% for Bibles. In contrast, books not tied to organized religions, such as ones on the subject of new age (2.5%) and occult (2.2%) spirituality, have lower recommendation effectiveness

There are exceptions of course. For example, Japanese anime DVDs have a strong following in the US, and this is reflected in their frequency and success in recommendations. Another example is that of gardening. In general, recommendations for books relating to gardening have only a modest chance of being accepted, which agrees with the individual prerogative that accompanies this hobby. At the same time, orchid cultivation can be a highly organized and social activity, with frequent ‘shows’ and online communities devoted entirely to orchids. Perhaps because of this, the rate of acceptance of orchid book recommendations is twice as high as those for books on vegetable or tomato growing.

## Products and recommendations
Since we do not have direct sales data we used the number of successful recommendations as a proxy to the number of **purchases**.
 Figure 15 plots the distribution of the number of purchases and the number of recommendations per product. Notice that both the number of recommendations and the number of purchases per product follow a heavy-tailed distribution and that the distribution of recommendations has a heavier tail
Interestingly, figure 15(a) shows that just the top 100 products account for 11.4% of the all sales (purchases with discount), and the top 1000 products amount to 27% of total sales through the recommendation system. On the other hand 67% of the products have only a single purchase and they account for 30% of all sales. This shows that a significant portion of sales come from products that sell very few times.
We also find that the tail is a bit longer than the usual 80-20 rule, with the top 20% of the products contributing to about half the sales.
![[figure_5.png]]

55% of the products have a success rate bellow 5% and there are around 14% of the products that have a recommendation success rate higher than 20%.
![[figure_6.png]]
##  what determines the product’s viral marketing success?
which characterizes product categories for which recommendations are more likely to be accepted. We use a regression of the following product attributes to correlate them with recommendation success:
• n: number of nodes in the social network (number of unique senders and receivers) 
• ns: number of senders of recommendations
• nr: number of recipients of recommendations 
• r: number of recommendations
•e: number of edges in the social network (number of unique (sender, receiver) pairs) 
• p: price of the product 
• v: number of reviews of the product 
• t: average product rating

From the original set of the half-million products, we compute a success rate s (CREDO SIA IL GROUND TRUTH) for the 8,192 DVDs and 50,631 books that had at least 10 recommendation senders and for which a price was given.
Since the variables follow a heavy tailed distribution, we use the following model:
![[figure_7.png]]
where xi are the product attributes (as described on previous page), and ǫi is random error. We fit the model using least squares and obtain the coefficients βi shown in table 8.
NOTA: fittato differenti modelli per BOOKS e DVDs
We find that the numbers of nodes and receivers have negative coefficients, showing that successfully recommended products are actually more likely to be not so widely popular.
The only attributes with positive coefficients are the number of recommendations r, number of edges e, and price p. (di conseguenza piu alto il prezzo piu success rate aumenta).
This shows that more expensive and more recommended products have a higher success rate. These recommendations should occur between a small number of senders and receivers, which suggests a very dense recommendation network where lots of recommendations are exchanged between a small community of people.
These insights could be of use to marketers — personal recommendations are most effective in small, densely connected communities enjoying expensive products.


To illustrate the dependencies between the variables we train a Bayesian dependency network [Chi03], and show the learned structure for the combined (Books and DVDs) data in figure 17. In this a directed acyclic graph where nodes are variables, and directed edges indicate that the distribution of a child depends on the values taken in the parent variables.
HERE IT IS
![[figure_8.png]]
Notice from this image that the average rating (t) is not predictive of the recommendation success rate (s). It is no surprise that the number of recommendations r is predictive of number of senders ns
Similarly, the number of edges e is predictive of number of senders ns. Interestingly, price p is only related to the number of reviews v. Number of recommendations r, number of senders ns and price p, are directly predictive of the recommendation success rate s


 IMPORTANTIUSSIMO METTILO
 Firstly, it is frequently assumed in epidemic models (e.g., SIRS type of models) that individuals have equal probability of being infected every time they interact [AM02, Bai75]. Contrary to this we observe that the probability of infection decreases with repeated interaction
Traditional epidemic and innovation diffusion models also often assume that individuals either have a constant probability of ‘converting’ every time they interact with an infected individual [GLM01], or that they convert once the fraction of their contacts who are infected exceeds a threshold [Gra78]. In both cases, an increasing number of infected contacts results in an increased likelihood of infection.

INSTEAD Instead, we find that the probability of purchasing a product increases with the number of recommendations received, but quickly saturates to a constant and relatively low probability.

In network-based epidemic models, extremely highly connected individuals play a very important role. For example, in needle sharing and sexual contact networks these nodes become the “super-spreaders” by infecting a large number of people. But these models assume that a high degree node has as much of a probability of infecting each of its neighbors as a low degree node does.

In contrast, we find that there are limits to how influential high degree nodes are in the recommendation network. As a person sends out more and more recommendations past a certain number for a product, the success per recommendation declines. This would seem to indicate that individuals have influence over a few of their friends, but not everybody they know.

We also presented a simple stochastic model that allows for the presence of relatively large cascades for a few products, but reflects well the general tendency of recommendation chains to terminate after just a short number of steps. Aggregating such cascades over all the products, we obtain a highly disconnected network, where the largest component grows over time by aggregating typically very small but occasionally fairly large components.

We saw that the characteristics of product reviews and effectiveness of recommendations vary by category and price, with more successful recommendations being made on technical or religious books, which presumably are placed in the social context of a school, workplace or place of worship.
...
altri punti che ho skippato
...

Since viral marketing was found to be in general not as epidemic as one might have hoped, marketers hoping to develop normative strategies for word-of-mouth advertising should analyze the topology and interests of the social network of their customers