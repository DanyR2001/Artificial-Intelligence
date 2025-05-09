# Gen 7
si nota che si è prese quelle che hanno superato r^2>0.8:
- 0_0_0_4000_40 -> R^2 0.8150
- 0_0_1_4000_40 -> R^2 0.8180

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 0_ [1024,512,256,128]

mentre il successivo numero sono i dropouts:
- 0 0.0
- 1 0.1

il penultimo indica la size dal dataset in input ovvero 4000 esempi e il successivo sono in numero di epoche ovvero 40

tutti le list:
sizeCSV=[250,500,750,1000,1500,2000]
hidden_sizes = [[1024,512,256,128],[1024,512,256,128,64]]
dropouts = [0.1, 0.2]
input_sizes = [768]
epochs = [10,20,30,40,50]