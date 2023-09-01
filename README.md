# Projeto de Machine Learning: Detecção de Notas Falsas em Imagens de Cédulas Bancárias

O projeto demonstra como usar o modelo para fazer predições sobre a autenticidade das notas bancárias e é uma aplicação valiosa da inteligência artificial em problemas de detecção de fraudes.


## Funcionalidades:

- O projeto começa importando as bibliotecas essenciais para a análise de dados e a criação de modelos de Machine Learning. Isso inclui bibliotecas como NumPy, pandas, Scikit-Learn, Seaborn, Matplotlib e Graphviz.
- Faz o download de um conjunto de dados (dataset) a partir de um URL usando comandos do Python. Em seguida, os dados são lidos em um DataFrame do Pandas para análise e processamento.
- Faz a divisão dos dados em um conjunto de treinamento e um conjunto de teste usando a função train_test_split do Scikit-Learn. Isso é fundamental para avaliar o desempenho do modelo.
- Cria um modelo de árvore de decisão usando o Scikit-Learn. Esse modelo é treinado com base nas características estatísticas das imagens das cédulas bancárias, como variância, assimetria, curtose e entropia.
- Utiliza métricas de avaliação, como o RMSE (Raiz Quadrada do Erro Quadrático Médio) e a acurácia. Isso permite determinar quão bem o modelo é capaz de distinguir notas bancárias genuínas de falsificadas.
- Oferece a opção de visualizar a árvore de decisão treinada usando a biblioteca Graphviz. Isso pode servir para entender como o modelo toma decisões com base nas características das cédulas bancárias.
Realiza a demonstração de como usar o modelo treinado para fazer predições sobre a autenticidade de uma cédula bancária com base em suas características estatísticas. Isso pode ser aplicado a novos dados para verificar se uma cédula é genuína ou falsificada.

## Como Usar

Faça o download do arquivo.ipynb: se você ainda não fez isso, baixe o arquivo do projeto, geralmente com a extensão.ipynb, para o seu computador.
Abra o Google Colab.
Carregue o arquivo.ipynb: no menu "Arquivo" do Google Colab, selecione a opção "Fazer upload de notebook". Em seguida, escolha o arquivo.ipynb que você baixou no passo anterior.

Importe as bibliotecas necessárias: No início do notebook, as bibliotecas necessárias já devem estar importadas. Verifique se as importações são as seguintes:

import numpy as np;
import pandas as pd;
import seaborn as sns;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, accuracy_score;
from sklearn import tree;
import graphviz

Execute o notebook: Clique em "Runtime" no menu superior e selecione "Run All". Isso executará todas as células do notebook, incluindo o download dos dados, a divisão dos dados, o treinamento do modelo e a avaliação do modelo.

Visualização da Árvore de Decisão (opcional): Se você quiser visualizar a árvore de decisão, certifique-se de que a célula com o código para a visualização da árvore não esteja comentada. Execute a célula para gerar a visualização.


## Observações

- Certifique-se de estar conectado à internet durante a execução, uma vez que o Google Colab exige a conexão com internet.
- Este projeto foi desenvolvido com propósitos educacionais, enfatizando a compreensão de Machine Learning com aprendizado supervisionado e Árvore de Decisão
- Contribuições e sugestões são bem-vindas para aprimorar este projeto. Sinta-se à vontade para criar problemas ou solicitações de incorporação.


Machine Learning Project: Detecting Fake Banknotes in Banknote Images
The project demonstrates how to use the model to make predictions about the authenticity of banknotes and is a valuable application of artificial intelligence in fraud detection problems.
Functionalities:
The project starts by importing the essential libraries for analyzing data and creating Machine Learning models. This includes libraries such as NumPy, pandas, Scikit-Learn, Seaborn, Matplotlib, and Graphviz.
It downloads a dataset from a URL using Python commands. The data is then read into a Pandas DataFrame for analysis and processing.
It splits the data into a training set and a test set using Scikit-Learn's train_test_split function. This is essential for evaluating the model's performance.
Create a decision tree model using Scikit-Learn. This model is trained based on the statistical characteristics of the banknote images, such as variance, asymmetry, kurtosis, and entropy.
It uses evaluation metrics such as RMSE (Root Mean Square Error) and accuracy. This allows you to determine how well the model can distinguish genuine banknotes from counterfeit ones.
It offers the option of visualizing the trained decision tree using the Graphviz library. This can be used to understand how the model makes decisions based on the characteristics of banknotes. It demonstrates how to use the trained model to make predictions about the authenticity of a banknote based on its statistical characteristics. This can be applied to new data to check whether a banknote is genuine or counterfeit.
How to use
Download the .ipynb file: If you haven't already done so, download the project file, usually with the extension .ipynb, to your computer. Open Google Colab. Upload the .ipynb file: in Google Colab's "File" menu, select the "Upload from the notebook" option. Then choose the .ipynb file you downloaded in the previous step.
Import the necessary libraries: At the start of the notebook, the necessary libraries should already be imported. Check that the imports are as follows:
import numpy as np; import pandas as pd; import seaborn as sns; from sklearn.tree import DecisionTreeClassifier; from sklearn.model_selection import train_test_split; from sklearn.metrics import mean_squared_error, accuracy_score; from sklearn import tree; import graphviz
Run the notebook: Click on "Runtime" in the top menu and select "Run All". This will run all the cells in the notebook, including downloading the data, splitting the data, training the model, and evaluating the model.
Visualizing the Decision Tree (optional): If you want to visualize the decision tree, make sure that the cell with the code for the tree visualization is not commented out. Run the cell to generate the visualization.
Notes
Make sure you are connected to the internet when running, as Google Colab requires an internet connection.
This project was developed for educational purposes, emphasizing the understanding of Machine Learning with supervised learning and Decision Tree
Contributions and suggestions are welcome to improve this project. Feel free to create issues or incorporation requests.
