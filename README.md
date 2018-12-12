# Hole-level-score-predictor

This project is a golf score predictor (herein GSP) for each hole played by each player in a PGA Tour tournament, using weather, player skill, and golf course hole difficulty as features in random forest and gradient boosted tree models. By one-hot encoding categorical features and using PCA to reduce the feature load, an accuracy and precision of 58% and a log-loss of 1.05 is achieved for the 2018 US Open at Shinnecock Hills.

For comparison, a model developed by Drappi and Key which uses data of each shot's distance and location - higher granularity compared to the hole-level data used in this GSP - achieved a log-loss of 0.891. Their data is spatially and temporally more precise, but due to their analysis of more tournaments - injecting randomness into a game that is inherently highly random - achieves an error that is incrementally lower. Furthermore, key features that were included in the GSP were left out of their analysis, such as wind direction, temperature, humidity, and pin placement difficulty. These features can change for each round of golf, drastically changing the field's average score on a particular hole.

Improvements to the GSP model should aim to 1) expand the feature set to include more course difficulty features and 2) include multiple PGA Tour tournaments from many years. The former is needed to account for the vast array of challenges presented to players by a variety of courses, and the latter is to train the model on more data from tournaments played on those courses. Unique course architectures might present players with predominantly water-lined fairways and tall rough one tournament, and a dry course with narrow fairways and elevated greens in another tournament. By expanding the sample size of holes played, the error of each score's predicted probability may be minimized.

A unique accomplishment of this project is that a respectable log-loss value is achieved with only publicly available data from websites.
