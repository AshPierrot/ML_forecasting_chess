This code works with a database of chess matches (a sample of over 20,000 games) to predict victory.
I saw a competition on Kaggle and, although I didn’t participate, I created a model for myself.
Its explanatory power - F1 measure, shows the quality of the model at 0.45 on the training group and 0.47 on the test group, respectively.
It is clear that predicting victory in a chess game is incredibly difficult, but interesting.
The distribution of games by rating is close to normal, which allows you to work with data using unique mathematical statistics tools
We predict victory in the game of one side or another (winner), depending on:
- Number of moves made (turns)
- End of game status (victory_status)
- Established rules (increment_code)
- White rating (white_rating)
- Black rating (black_rating)
- Debut code (opening_eco)
- Debut Name (opening_name)
- Number of theoretical moves (opening_ply)
