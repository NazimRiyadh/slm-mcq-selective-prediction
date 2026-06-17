# Verified broad non-MCQ final results audit

Used validation ZIP: `slm_locked_validation_outputs_q1_broad_nonmcq.zip`.

Discovered conditions: 38 total.

Dataset counts:
dataset
arc_challenge    7
commonsenseqa    7
hellaswag        7
mmlu_all         5
agnews           3
trec6            3
dbpedia14        3
mmlu             2
banking77        1

Primary locked rows: 510 = 34 conditions x 5 seeds x 3 selector families.

Main MCQ conditions: 24.
Non-MCQ finite-choice conditions: 9.
All auxiliary extension conditions including Banking77: 10.

The second uploaded ZIP `slm_locked_validation_outputs (1).zip` was not used because it contains only the older 28 MCQ conditions and lacks AG News, TREC, DBPedia, and Banking77.

Main MCQ locked means:
                       test_auroc_correct  test_auprc_failure  test_risk_at_80  test_aurc
family                                                                                   
cheap_plus_hidden_pca            0.781953            0.558076         0.191580   0.119884
confidence_option                0.798848            0.570475         0.189913   0.115299
hidden_pca                       0.707141            0.468290         0.212604   0.152034

Non-MCQ locked means:
                       test_auroc_correct  test_auprc_failure  test_risk_at_80  test_aurc
family                                                                                   
cheap_plus_hidden_pca            0.865911            0.679745         0.151035   0.092558
confidence_option                0.857355            0.653117         0.154177   0.095972
hidden_pca                       0.853729            0.641607         0.155857   0.097771

All primary locked means:
                       test_auroc_correct  test_auprc_failure  test_risk_at_80  test_aurc
family                                                                                   
cheap_plus_hidden_pca            0.805600            0.597967         0.185323   0.115670
confidence_option                0.814161            0.597944         0.185897   0.113410
hidden_pca                       0.749611            0.523935         0.202359   0.139319
