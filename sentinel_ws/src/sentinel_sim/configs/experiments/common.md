Evaluation protocol shared by all experiment files (REWORK_PLAN §5, §6):
- seeds 0-29 (evaluation); tuning used 1000+, MAPPO training 2000+
- tuned hyperparameters loaded from ../tuned.yaml for every method (equal budget)
- default scenario: 400 m map, 25 victims (1 buried), responder at the launch corner,
  urban_rubble channel, Poisson failures 0.01/s, 600 s mission
