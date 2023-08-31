This folder contains two demoes.
The first demo is Fine-Tuning demo. I use the pre-train ResNet-18 based ImageNet1K and change the full connect layer from 1000 to 2 and train it at our dataset. Our dataset contains 2 classes, Bees and Ants.
The second demo is the linearProb. I use the pre-train ResNet-18 based ImageNet1K and freeze all the layers expect the final layer. Then, we transfer it to our dataset.

The dataset link: https://drive.google.com/file/d/1Yaei-8g41aBxm4Htwu9AWYMZDpRm6l1p/view?usp=drive_link
