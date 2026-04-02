Look at data you have for 2026. get that for the past. We don't use RPI in 2026 so who cares


- Why do we even need Warner Nolan team sheet? Use all other data first, then come back to it
- move all features into kaggle datasets and just build on top of their dataset.
    - [ ] SOS 
        - (from nitty gritty for 2020-2026 then pdf from before 2020)
    - [ ]  BPI
        - We ONLY put 2026 bpi from espn bc it is pre F4. get bpi from team sheets for before that that. bpi is is 09-13 (kaggle) and 18-19 (rpi team sheets) and 26 (espn) missing for 2021-2025. so only  
        - CHECK: are we just using BPI to see finals (so like <10 seasons of data?) or is it for every game (fo game n, check bpi for both team and see predictive power; more data this way but bpi early season might be weak predictively)
    - [ ] SOR
        - it is in the rpi archive but not nitty gritty. get it from espn or warren nolan
    - [ ] KPI
        - from warren nolan or official source
    - [ ] POM
        - from warren nolan
    - [ ] WAB
        - from warren nolan 
    - [ ] SOR (get from espn lowkey; but also in warren nolan) 

[ ] Figure out how final four data is used to test features. We don't want lookahead bias. The question we are tying to solve is: Given a final 4 and past data, who will likely be/not be the champion? Does the model do random walks (ie. given A vs B and C vs D in final four, what happens if A vs C or A vs D or ... in finals what are the chances of A winning? they can lose against B,C, or D)
    - I think this is done via pairwise
    - NOTE: the feature testing should do ALL pair wise comparisons using a rolling window (ie. at game 0 there is no prior so just wild guess, but at game n...) to see which features are really important rather than just doing it at the final four for each season. does that sound good or bad?
    - IN FACT: i think we should refactor the model featuring from "Who will win the final four" to "who will win this game" and use baysian stats since we have more data on games than just finals since finals is a subset of all games. THis makes sense, because the final four is just one random walk, but we are still only fouxed on the final 4 soooo...

[!] I don't think W-L ratios are in the features
    - same with avg appoinent net rank, opp SOS, oppoenent W-L, which can all be calculated from the raw nitty gritty or kaggle
    - there is a nother feature i added that i want to test

[!] Add these as features: PrevNET,AvgOppNETRank,AvgOppNET,NET margin can be found in nitty gritty csvs except net margin can be done by mapping the teams to their net in kaggle data MM ordinal csv and Mteam csv
    - avg app net can be done using on the spot calculation (sum of net- yours)/n and not from getting the column in nitty gritty. the only transferable columns are NETSOS,NETNonConfSOS

[!] the feature pipeliene let past 5 days and 3 days of score margin come pass but they are overlapping data why is it passing

[ ] add the afeature of away games won and performance of away games (sum of avg weighted stats during home games, like score marge gets a heavy weight and steals get a light weight and turnorvers get a heavy weight) since finals are technically away games for both teams

[ ] add feature of 