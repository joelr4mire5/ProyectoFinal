import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prince import PCA as PCA_Prince
class ACP:
    def __init__(self, datos, n_componentes = 5):
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations(datos)
        self.__coordenadas_ind = self.__modelo.row_coordinates(datos)
        self.__contribucion_ind = self.__modelo.row_contributions(datos)
        self.__cos2_ind = self.__modelo.row_cosine_similarities(datos)
        self.__var_explicada = [x * 100 for x in self.__modelo.explained_inertia_]
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
        self.__var_explicada = var_explicada
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, c=["steelblue"],
            edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('tab10', 150))
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue',
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i],
                         color = 'steelblue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True,
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))),
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'steelblue')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue',
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.correlacion_var.index[i],
                         color = 'steelblue', ha = 'center', va = 'center')



class ACP2(ACP):
    def __init__(self, datos, n_componentes=5, cols=[]):
        ACP.__init__(self, datos[cols], n_componentes)

    def plot_plano_principal(self, ejes=[0, 1], ind_labels=True, titulo='Plano Principal', cos=0.10):

        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cosx = self.cos2_ind[0].reset_index(drop=True)
        cosy = self.cos2_ind[1].reset_index(drop=True)
        Data = pd.DataFrame(x)

        Data.insert(1, 1, y, True)
        Data.insert(2, "Cosx", cosx, True)
        Data.insert(3, "Cosy", cosy, True)
        Data = Data[Data["Cosx"] > cos]
        Data = Data[Data["Cosy"] > cos]
        x = Data[0].values
        y = Data[1].values

        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color='steelblue')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(Data.index):
                plt.annotate(txt, (x[i], y[i]))

    def plot_sobreposicion(self, ejes=[0, 1], ind_labels=True,
                           var_labels=True, titulo='Sobreposición Plano-Círculo', cos=0.10):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values

        cosx = self.cos2_ind[0].reset_index(drop=True)
        cosy = self.cos2_ind[1].reset_index(drop=True)
        Data = pd.DataFrame(x)

        Data.insert(1, 1, y, True)
        Data.insert(2, "Cosx", cosx, True)
        Data.insert(3, "Cosy", cosy, True)
        Data = Data[Data["Cosx"] > cos]
        Data = Data[Data["Cosy"] > cos]
        x = Data[0].values
        y = Data[1].values

        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x) / (max(cor[ejes[0]]) - min(cor[ejes[0]]))),
                    (max(y) - min(y) / (max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color='steelblue')
        if ind_labels:
            for i, txt in enumerate(Data.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')



Le = LabelEncoder
scaler = StandardScaler


df= pd.read_csv("Input/marketing_campaign.csv",sep='\t')

df.drop(columns=['ID','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2','NumDealsPurchases','Response','Dt_Customer'],inplace=True)

df['Age']=2021-df['Year_Birth']
df.drop(columns="Year_Birth", inplace=True)

df["Education"]= Le().fit_transform(df["Education"])
df["Marital_Status"]= Le().fit_transform(df["Marital_Status"])
df.dropna(inplace=True)
print(df.columns)

df_scaled=scaler().fit_transform(df)

data=pd.DataFrame(df_scaled,columns=df.columns)


acp2 = ACP2(data,n_componentes=2,cols=['Education', 'Marital_Status','Income', 'Kidhome', 'Teenhome',
       'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Age'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,12), dpi = 200)
acp2.plot_plano_principal(ejes = [0, 1],cos=0.10)
plt.show()



fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,12), dpi = 200)
acp2.plot_sobreposicion(ejes = [0, 1],cos=0.10)
plt.show()




