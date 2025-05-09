# Gen 3
si nota che molte iterazioni hanno superato un r^2>0.79 ovvero 11/21 di conseguenza si è prese quelle che hanno superato r^2>0.8:
- 0_0_0 -> R^2 0.8028
- 0_1_0 -> R^2 0.8025
- 0_4_0 -> R^2 0.8029

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 0_ [1024,512,256,128]
- 1_ [1024,512,256,128,64]
- 4_ [512,128,64]

mentre l'ultimo numero sono i dropouts:
- 0_ 0.1

Quindi provvederemo a rimuovere le altre input size, il dropout=0.4 e si creeranno altri valori per l'hidden_size
tutti le list:
hidden_sizes=[[1024,512,256,128],[1024,512,256,128,64],[512, 256, 124],[512, 256, 128],[512,128,64],[512,124],[512,128]]
dropouts=[0.1,0.2,0.3]
input_sizes=[768]