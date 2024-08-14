---
layout: page
title: Second Price Auction Simulator
description: Object Oriented Programming with Python
img: assets/img/auction.jpg
importance: 3
category: Python Programming
---

This project was an implementation of creating a `Bidder` class that attempts to create the best bidder class in a [second-price auction](https://smartclip.tv/adtech-glossary/second-price-auction/) to show website users an advertisement. The idea of this was to utilize the multi-armed bandit problem in order to create a `Bidder` that balances exploration and exploitation in order to get the user's attention as much as possible.

We will walk through the implementation by first going over the `Auction` class. The `Auction` will take inputs of `users` and `bidders`, which are lists that consists of different `User` and `Bidder` class objects, respectively. It will then initialize the variables and also keep track of the balances of each of the `bidders`.

{% raw %}

```python
class Auction:
    '''Class to represent an online second-price ad auction'''
    def __init__(self, users, bidders):
        '''Initializing users, bidders, and dictionary to store balances 
        for each bidder in the auction'''
        self.users = users
        self.bidders = bidders.copy()
        self.balances = {bidder: 0 for bidder in self.bidders}
```

{% endraw %}

The below code is how the `Auction` executes a round. It begins by choosing a random user to show all the `bidders`. Based on the random `user_id`, they will output a certain valued `bid`. The auction will then choose the `Bidder` as the one who bid the most money and will then determine what the next highest price was, which we will call `winning_price`. The `winning_price` will be the amount of money that the winning bidder will have to pay in order to show the user their ad. The `Auction` will then show a user an advertisement to which the user will output a boolean for whether they `clicked` on the ad or not. Finally, the auction will find the winning `bidder`, add one dollar to their balance if the user `clicked` on the ad, and subtract the `winning_price` for the round. Even if a particular `Bidder` did not win, it will still be notified that it did not win and will inform them what the `winning_price` was. 

{% raw %}

```python
    def execute_round(self):
        '''Executes a single round of an auction, completing the following steps:
            - random user selection
            - bids from every qualified bidder in the auction
            - selection of winning bidder based on maximum bid
            - selection of actual price (second-highest bid)
            - showing ad to user and finding out whether or not they click
            - notifying winning bidder of price and user outcome and updating balance
            - notifying losing bidders of price'''
        user_id = np.random.randint(0,len(self.users)) # choose a random user_id for the Auction
        winning_indices = []
        for index,bidder in enumerate(self.bidders):
            if self.balances[bidder] < -1000: # remove bidders that have less than -1000 dollars
                del self.balances[bidder]
                del self.bidders[index]
        if len(self.balances) == 0: #end Auction if there are no more bidders remaining
            return 'All Bidders are disqualified' 
        
        # show all the bidders the user_id and request for them to bid
        bids = [bidder.bid(user_id) for index,bidder in enumerate(self.bidders)]
        max_price = max(bids) # find the highest bid

        for index,bid in enumerate(bids): # find the bidder with the highest bid
            if bid == max_price:
                winning_indices.append(index)
        
        if len(winning_indices) > 1:  # if there is more than one bidder that had the highest bid, then randomly choose one of the indices
            winning_index = np.random.choice(winning_indices)
        else:
            winning_index = winning_indices[0]
        
        if len(bids) > 1: # find the second highest bid, which is the price that the winning bidder is going to have to pay
            remaining_values = bids[:winning_index] + bids[winning_index+1:]
            winning_price = max(remaining_values)
        else:
            winning_price = max(bids)

        clicked = self.users[user_id].show_ad() # show the user the ad and determine whether they clicked on the ad or not

        for index,bidder in enumerate(self.bidders):
            if index == winning_index: # if the bidder won the round, then notify the bidder and update their balance
                bidder.notify(True,winning_price,clicked)
                self.balances[bidder] = self.balances[bidder] \
                                        + int(clicked) - winning_price # if the user clicked the ad then +1 dollar but remove the price of the second highest bid
            else:
                bidder.notify(False,winning_price,None) # if the bidder did not win, then just notify the bidder of the winning price
```

{% endraw %}

We now move onto the `User` class, now knowing how the `Auction` class works. Fortunately, the `User` class is much simpler to understand. Each `User` class that is created will be assigned a random probability variable that is just a value pulled from a uniform distribution from [0,1]. When the `User` is shown an ad in the `show_ad()` function, they will choose a boolean value that represents whether they will click on the ad or not, based on their `__probability` variable. 

{% raw %}

```python

class User:
    '''Class to represent a user with a secret probability of clicking an ad.'''
    def __init__(self):
        '''Generating a probability between 0 and 1 from a uniform distribution'''
        self.__probability =  np.random.uniform()
    def __repr__(self):
        '''User object with secret probability'''
        return f'This user has a probability {self.__probability} of clicking on an ad'
    def __str__(self):
        '''User object with a secret likelihood of clicking on an ad'''
        return f'This user has a probability {self.__probability} of clicking on an ad'
    def show_ad(self):
        '''Returns True to represent the user clicking on an ad or False otherwise'''
        return np.random.choice([True,False],p=[self.__probability , 1 - self.__probability])

```

{% endraw %}

The bidders is where we will modify the strategy based on which `User` they're bidding on and how much they should bid. The simplest `Bidder` class is the `RandomBidder`, shown in the code below. We first initialize it with the number of users, and number of rounds that are in the `Auction`. The `bid()` method for the `RandomBidder` just returns a random float value from a uniform distribution from [0,1]. When it is notified through the `notify()` method, it will then update it's internal balance. If it won the bid, we will always subtract what the `price` was from the round. If it won the bid and the `User` clicked on it's ad, then we will increase the balance by one dollar. For all the `Bidder` classes that we create, we will also incorborate a `self.balances_over_time`, which is a list that keeps track of how the balances of each object changes with the number of rounds. 

{% raw %}

```python
class RandomBidder:
    '''Class to represent a Random bidder in an online second-price ad auction'''
    
    def __init__(self, num_users=0, num_rounds=0):
        '''Setting initial balance to 0, number of users, number of rounds, and round counter'''
        self.num_users = num_users
        self.num_rounds = num_rounds
        self.clicked = None
        self.winning_price = 0
        self.balance = 0
        self.balances_over_time = [0]

    def bid(self, user_id):
        '''Returns a non-negative bid amount'''
        bids = round(np.random.uniform(0,1),3) # output a random bid amount
        return bids

    def notify(self, auction_winner, price, clicked):
        '''Updates bidder attributes based on results from an auction round'''
        if auction_winner: # if it won the auction, subtract the second highest price from its balance
            self.balance -= round(price,3)
            if clicked: # if the user clicked on the ad, then add one dolalr to balance
                self.balance += 1
            self.balances_over_time.append(self.balance) # append the new balance
            
        else: # if it did not win the Auction round, just update the current balance for the round
            self.winning_price = price
            self.balances_over_time.append(self.balance)
```
{% endraw %}

The `EpsilonGreedyBidder` is the next iteration from the `RandomBidder` class, shown in the code below. The main difference with this class is that we now keep track of whether this bidder won the auction round and with which user. In this case, we can then create an estimate for each users probability. We will initialize an `epsilon` value, which is a float value from 0 to 1. In the `bid()` method, the `Bidder` will choose a random float value from 0 to 1, if it's higher than the `epsilon` threshold, the `Bidder` will then look at the current `user_prob`, which is the estimated probability that the user will click based on past rounds. If the `user_prob` is the highest probability, then then `Bidder` will bet 0.99 dollars, or the maximum amount without actually losing money as the most amount of money that a `Bidder` will gain is $1 only if the `User` clicks on the add. Finally, after a round ends, the `notify()` function will determine whether the `Bidder` won or not. If the `Bidder` won, then the user probability will be updated for that specific user based on whether they clicked or not. 

{% raw %}

```python
class EpsilonGreedyBidder:
    '''Class to represent an Epsilon Greedy bidder in an online second-price ad auction'''
    
    def __init__(self, num_users=0, num_rounds=0, epsilon=0.1):
        '''Setting initial balance to 0, number of users, number of rounds, and round counter'''
        self.num_rounds = num_rounds
        self.num_users = num_users
        self.winning_prices = [0 for i in range(num_rounds)]
        self.user_clicked = [0 for i in range(num_users)]
        self.user_won_auction = [0 for i in range(num_users)]
        self.user_prob = [0 for i in range(num_users)]
        self.balance = 0
        self.epsilon = epsilon
        self.balances_over_time = [0]

    def bid(self, user_id):
        '''Returns a non-negative bid amount'''
        self.current_user_id = user_id
        if np.random.uniform() < self.epsilon:
            bids = round(np.random.uniform(),3)
        else:
            if self.user_prob[self.current_user_id] == max(self.user_prob):
                bids = 0.99
            else:
                bids = round(np.random.uniform(0,0.99),3)
                
        self.bids_over_time.append(bids)
        return bids

    def notify(self, auction_winner, price, clicked):
        '''Updates bidder attributes based on results from an auction round'''
        if auction_winner:
            self.user_won_auction[self.current_user_id] += 1
            
            if clicked:
                self.balance += 1 - price
                self.user_clicked[self.current_user_id] += 1 - price
            else:
                self.balance -= price
                self.user_clicked[self.current_user_id] -= price
                
            self.user_prob[self.current_user_id] = self.user_clicked[self.current_user_id]/self.user_won_auction[self.current_user_id]
        
        self.balances_over_time.append(self.balance)
```
{% endraw %}

Finally, we attempt to incorporate the Upper Confidence Bound (UCB) method into our `Bidder` class. The mechanics for how UCB works is shown [here](https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/); however, the formula we implement is shown below, where A<sub>t</sub> is the action that the `Bidder` wants to take, Q<sub>t</sub>(a) is a conditional expectation, or the expected reward from taking an action a. We will estimate it by determining how much money the `Bidder` has been rewarded based on the number of times that it has bid a high amount. The last term, labelled as `explore` in the picture below, adds a certain amount of uncertainty to the equation. It is based on the number of rounds and the number of times that we have seen this `User`. If there have been very few rounds, then the explore term will be higher and the `Bidder` will be more likely to bid a high amount. On the other hand, if the `Auction` has been going on for awhile, theoretically, the `explore` term will decrease and the `Bidder` will take action based on it's prior knowledge.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ucb.jpg" title="graph layout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Formula for Upper Confidence Bound Method (UCB)
</div>

{% raw %}

Knowing this formula, we implement it into our `UCBBidder` class with the code below. Not only do we keep track of our user's proabilities through the `user_prob` list; however, we also keep track of the number of rounds (`self.round_counter`) and the number of times that the `Bidder` has won an auction with this specific user (in `self.user_won_auction`). The general procedure for the `.bid()` function will be as follows:

1. The `Bidder` will see if we have won with that specific `User` at least 10 times. If it has been less than 10 times, we automatically want to explore with that user by bidding the highest bid of $1 to gain more information about the `User's` probability of clicking.
2. If we have won with that `User` more than 10 times, we will check to see if that user has the highest `ucb` score in the list of `self.user_ucb`. If it does, then the `Bidder` will bid the maximum amount that will still gain the `Bidder` money (0.99 dollars)
3. If the `User` does not have the highest `ucb` score, then the `Bidder` will bet a random amount to still learn more about the bidder.

```python
class UCBBidder:
    '''Class to represent a UCB bidder in an online second-price ad auction'''    
    def __init__(self, num_users=1, num_rounds=1):
        '''Setting initial balance to 0, number of users, number of rounds, and round counter'''
        self.winning_prices = [0 for i in range(num_rounds)]
        self.user_profit = [0 for i in range(num_users)]
        self.user_won_auction = [0 for i in range(num_users)]
        self.user_ucb = [0 for i in range(num_users)]
        self.user_prob = [0 for i in range(num_users)]
        self.balances_over_time = [0]
        self.balance = 0
        self.current_user_id = ''
        self.round_counter = 0

    def bid(self, user_id):
        '''Returns a non-negative bid amount. The bid amount will be based on a variation of the
        Upper Confidence Bound (UCB) Method'''
        self.current_user_id = user_id
        if self.user_won_auction[user_id]<10:
            bids = 1
        elif self.user_won_auction[user_id] >= 10:
            if self.user_ucb.index(max(self.user_ucb)) == user_id:
                bids = 0.99
            else:
                bids = round(np.random.uniform(0,self.user_prob[self.current_user_id]),3)
        return bids
    def notify(self, auction_winner, price, clicked):
        '''Updates bidder attributes based on results from an auction round'''
        self.winning_prices.append(price)
        #Will keep a track of the bidder's balance whether they won or not.
        if auction_winner:
            self.user_won_auction[self.current_user_id] += 1
            if clicked:
                self.balance += 1 - price
                self.user_profit[self.current_user_id] += 1 - price
                self.round_counter += 1
            else:
                self.balance -= price
                self.user_profit[self.current_user_id] -= price
                self.round_counter += 1
            #Find new probability of the user by dividing profit over number of times seen with user
            self.user_prob[self.current_user_id] = self.user_profit[self.current_user_id]/self.user_won_auction[self.current_user_id]
            #Use UCB1 Formula to calculate the new user's likelihood of betting more
            self.user_ucb[self.current_user_id] = self.user_prob[self.current_user_id] + math.sqrt((2*math.log(self.round_counter) / (self.user_won_auction[self.current_user_id])))
        self.balances_over_time.append(self.balance)
```
{% endraw %}

The `notify()` function now works slightly differently where we now keep track of the profit that the `Bidder` gains from the Auction, if the user clicked or not in order to calculate the Q<sub>t</sub>(a) function. The profit is just 1 - the winning price fromt the auction. The profit is then divided by the number of times the bidder has won with that user (whether the `User` clicked or not). This will then be used to update the `self.user_ucb` for the `user_id` that they are bidding on. 

Now we have our `Auction`, our `Bidders`, and our `User` so we can simulate an auction with different numbers of `Users` and different `Bidders`. I created multiple `EpsilonGreedyBidders` in order to see the effect of changing `epsilon` in 0.1 increments. The results of the Auction are shown below for an Auction that lasted 10,000 rounds and with 5 `Users`. Overall, we can see that the `UCBBidder` does far better than any of the other `Bidders` after around 2000 rounds. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/auction_sim_05.png" title="graph layout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simulation of 10000 rounds of an Auction with n = 5 users
</div>

If we instead increase the number of `Users` to 100, then we see where the UCB method drops off significantly. The reason for this is how we implemented the bidding function, where we attempt to explore as much as possible and want to try and win with each `User` at least 10 times before considering their `ucb` score.  

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/auction_sim_100.png" title="graph layout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simulation of 10000 rounds of an Auction with n = 100 users
</div>

Ultimately, there are a lot more modifications that can be made with each of these models (especially with trying to output a positive balance); however, this was a great introduction into the Multi-Armed Bandit Problem and the online Advertising space. This project revealed how Object Oriented Programming can be utilized to simulate events and how to create different variations of the same classes from it. 