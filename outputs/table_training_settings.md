| Dataset | Blocks C | Table | Dropout | Weight decay | Negatives | Partner updates |
|---|---|---|---|---|---|---|
| Wikipedia | 96 equal | no | 0 | -- | seen | no |
| Reddit | 48 equal | no | 0 | -- | seen | no |
| MOOC | 192 equal | no | 0 | -- | seen | no |
| LastFM | 192 equal | no | 0 | -- | seen | no |
| Enron | 192 equal | no | 0 | -- | seen | yes |
| Social Evo. | 96 equal | no | 0 | -- | seen | no |
| UCI | 192 equal | yes | 0.2 | 0 | all | no |
| Flights | step | yes | 0 | 0.01 | all | no |
| Can. Parl. | step | no | 0 | -- | seen | no |
| US Legis. | step | yes | 0.2 | 0 | all | no |
| UN Trade | step | no | 0 | -- | seen | no |
| UN Vote | step | no | 0 | -- | seen | no |
| Contact | 192 equal | no | 0 | -- | seen | yes |

Shared by every stream: d=128, r=16, 256 random projection dimensions, M=20, learning rate 10^{-3}, at most 500 epochs, early stopping after 100 epochs without an improvement of the validation AP and AUC (50 on LastFM and Flights), five seeds.
