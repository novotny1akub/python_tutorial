# List Comprehension for df with zip https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas 
# list comprehension vs np.select
# np.select vs df.apply(lambda row
# data types Python and/or Pandas
# konverzni funkce
# virtualni prostredi
# debugging
# list dir

# Visualization: plotnine
# Time series / date functionality
# Dash/Bokeh/JupyterLab dashboarding
# datetime dle "relative products"

# utilities ---------------------------------------------------------------
# remove all variables from global
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Python Programming ------------------------------------------------------
# https://learnxinyminutes.com/docs/cs-cz/python/

####################################################
## 0. String manipulation
####################################################
word = "Hello World"
len(word)
word.count('l')
word.find("H")
word.index("World")
word.count(' ')
word[:3]
word.split(' ')
word.startswith("H")
word.endswith("w")
word.replace("Hello", "Goodbye")
word.upper()
word.lower()
word.title()
"hello world".capitalize() # 'Hello world'
word.swapcase()
' '.join(reversed(word))
"  hello     world ".strip()

word.isalnum() #check if all char are alphanumeric 
word.isalpha() #check if all char in the string are alphabetic
word.isdigit() #test if string contains digits
word.istitle() #test if string contains title words
word.isupper() #test if string contains upper case
word.islower() #test if string contains lower case
word.isspace() #test if string contains spaces

####################################################
## 1. Primitivní datové typy a operátory
####################################################

# Až na dìlení, které vrací desetinné èíslo
35 / 5  # => 7.0
# Pøi celoèíselném dìlení je desetinná èást oøíznuta (pro kladná i záporná èísla)
5 // 3       # => 1
5.0 // 3.0   # => 1.0  #  celoèíselnì dìlit lze i desetinným èíslem
-5 // 3      # => -2
-5.0 // 3.0  # => -2.0
# Pokud použijete desetinné èíslo, výsledek je jím také
3 * 2.0  # => 6.0
# Modulo
7 % 3  # => 1
# Mocnìní (x na y-tou)
2**4  # => 16

# Používání logických operátorù s èísly
0 and 2     # => 0
-5 or 0     # => -5

# Pøi porovnání s boolean hodnotou nepoužívejte operátor rovnosti "==". 
# Stejnì jako u hodnoty None.
# Viz PEP8: https://www.python.org/dev/peps/pep-0008/ 
0 is False  # => True
2 is True   # => False
1 is True   # => True
"nìco" is None  # => False
None is None    # => True
# None, 0, a prázdný øetìzec/seznam/N-tice/slovník/množina se vyhodnotí jako False
# Vše ostatní se vyhodnotí jako True
bool(0)        # => False
bool("")       # => False
bool([])       # => False
bool(tuple())  # => False
bool({})       # => False
bool(set())    # => False

# Øetìzec lze považovat za seznam znakù
"Toto je øetìzec"[0]  # => 'T'

# .format lze použít ke skládání øetìzcù
"{} mohou být {}".format("øetìzce", "skládány")
# Formátovací argumenty mùžete opakovat
"{0} {1} støíkaèek støíkalo pøes {0} {1} støech".format("tøi sta tøicet tøi", "støíbrných")
# Pokud nechcete poèítat, mùžete použít pojmenované argumenty
"{jmeno} si dal {jidlo}".format(jmeno="Franta", jidlo="guláš")  # => "Franta si dal guláš"

####################################################
## 2. Promìnné a kolekce
####################################################
# Seznam (je kolekce) se používá pro ukládání sekvencí
sez = []
# Lze ho rovnou naplnit
jiny_seznam = [4, 5, 6]
# Na konec seznamu se pøidává pomocí append
sez.append(1)    # sez je nyní [1]
sez.append(2)    # sez je nyní [1, 2]
sez.append(4)    # sez je nyní [1, 2, 4]
sez.append(3)    # sez je nyní [1, 2, 4, 3]
# Z konce se odebírá se pomocí pop
sez.pop()        # => 3 a sez je nyní [1, 2, 4]
# Vložme trojku zpátky
sez.append(3)    # sez je nyní znovu [1, 2, 4, 3]
# Pøístup k prvkùm funguje jako v poli
sez[0]  # => 1
# Mínus poèítá odzadu (-1 je poslední prvek)
sez[-1]  # => 3
# Pøístup mimo seznam vyhodí IndexError
sez[4]  # Vyhodí IndexError
# Pomocí øezù lze ze seznamu vybírat rùzné intervaly
# (pro matematiky: jedná se o uzavøený/otevøený interval)
sez[1:3]  # => [2, 4]
# Odøíznutí zaèátku
sez[2:]  # => [4, 3]
# Odøíznutí konce
sez[:3]  # => [1, 2, 4]
# Vybrání každého druhého prvku
sez[::2]  # =>[1, 4]
# Vrácení seznamu v opaèném poøadí
sez[::-1]  # => [3, 4, 2, 1]
# Lze použít jakoukoliv kombinaci parametrù pro vytvoøení složitìjšího øezu
# sez[zacatek:konec:krok]
# Odebírat prvky ze seznamu lze pomocí del
del sez[2]   # sez je nyní [1, 2, 3]
# Seznamy mùžete sluèovat
# Hodnoty sez a jiny_seznam pøitom nejsou zmìnìny
sez + jiny_seznam   # => [1, 2, 3, 4, 5, 6]
# Spojit seznamy lze pomocí extend
sez.extend(jiny_seznam)   # sez je nyní [1, 2, 3, 4, 5, 6]
# Kontrola, jestli prvek v seznamu existuje, se provádí pomocí in
1 in sez  # => True
# Délku seznamu lze zjistit pomocí len
len(sez)  # => 6


# N-tice je jako seznam, ale je nemìnná
ntice = (1, 2, 3)
ntice[0]      # => 1
ntice[0] = 3  # Vyhodí TypeError
# S n-ticemi lze dìlat vìtšinu operací, jako se seznamy
len(ntice)         # => 3
ntice + (4, 5, 6)  # => (1, 2, 3, 4, 5, 6)
ntice[:2]          # => (1, 2)
2 in ntice         # => True
# N-tice (nebo seznamy) lze rozbalit do promìnných jedním pøiøazením
a, b, c = (1, 2, 3)  # a je nyní 1, b je nyní 2 a c je nyní 3
# N-tice jsou vytváøeny automaticky, když vynecháte závorky
d, e, f = 4, 5, 6
# Prohození promìnných je tak velmi snadné
e, d = d, e  # d je nyní 5, e je nyní 4


# Slovníky ukládají klíèe a hodnoty
prazdny_slovnik = {}
# Lze je také rovnou naplnit
slovnik = {"jedna": 1, "dva": 2, "tøi": 3}
# Pøistupovat k hodnotám lze pomocí []
slovnik["jedna"]  # => 1
# Všechny klíèe dostaneme pomocí keys() jako iterovatelný objekt. Nyní ještì
# potøebujeme obalit volání v list(), abychom dostali seznam. To rozebereme
# pozdìji. Pozor, že jakékoliv poøadí klíèù není garantováno - mùže být rùzné.
list(slovnik.keys())  # => ["dva", "jedna", "tøi"]
# Všechny hodnoty opìt jako iterovatelný objekt získáme pomocí values(). Opìt
# tedy potøebujeme použít list(), abychom dostali seznam. Stejnì jako
# v pøedchozím pøípadì, poøadí není garantováno a mùže být rùzné
list(slovnik.values())  # => [1, 2, 3]
# Operátorem in se lze dotázat na pøítomnost klíèe
"jedna" in slovnik  # => True
1 in slovnik        # => False
# Pøístup k neexistujícímu klíèi vyhodí KeyError
slovnik["ètyøi"]  # Vyhodí KeyError
# Metoda get() funguje podobnì jako [], ale vrátí None místo vyhození KeyError
slovnik.get("jedna")   # => 1
slovnik.get("ètyøi")   # => None
# Metodì get() lze pøedat i výchozí hodnotu místo None
slovnik.get("jedna", 4)   # => 1
slovnik.get("ètyøi", 4)   # => 4
# metoda setdefault() vloží prvek do slovníku pouze pokud tam takový klíè není
slovnik.setdefault("pìt", 5)  # slovnik["pìt"] je nastaven na 5
slovnik.setdefault("pìt", 6)  # slovnik["pìt"] je poøád 5
# Pøidání nové hodnoty do slovníku
slovnik["ètyøi"] = 4
# Hromadnì aktualizovat nebo pøidat data lze pomocí update(), parametrem je opìt slovník
slovnik.update({"ètyøi": 4})  # slovnik je nyní {"jedna": 1, "dva": 2, "tøi": 3, "ètyøi": 4, "pìt": 5}
# Odebírat ze slovníku dle klíèe lze pomocí del
del slovnik["jedna"]  # odebere klíè "jedna" ze slovnik


# Množiny ukládají ... pøekvapivì množiny
prazdna_mnozina = set()
# Také je lze rovnou naplnit. A ano, budou se vám plést se slovníky. Bohužel.
mnozina = {1, 1, 2, 2, 3, 4}  # mnozina je nyní {1, 2, 3, 4}
# Pøidání položky do množiny
mnozina.add(5)  # mnozina je nyní {1, 2, 3, 4, 5}
# Prùnik lze udìlat pomocí operátoru &
jina_mnozina = {3, 4, 5, 6}
mnozina & jina_mnozina  # => {3, 4, 5}
# Sjednocení pomocí operátoru |
mnozina | jina_mnozina  # => {1, 2, 3, 4, 5, 6}
# Rozdíl pomocí operátoru -
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}
# Operátorem in se lze dotázat na pøítomnost prvku v množinì
2 in mnozina  # => True
9 in mnozina  # => False


####################################################
## 3. Øízení toku programu, cykly
####################################################

x = 0
while x < 4:
    print(x)
    x += 1  # Zkrácený zápis x = x + 1. Pozor, žádné x++ neexisuje.
    

# Výjimky lze ošetøit pomocí bloku try/except(/else/finally)
try:
    # Pro vyhození výjimky použijte raise
    raise IndexError("Pøistoupil jste k neexistujícímu prvku v seznamu.")
except IndexError as e:
    print("Nastala chyba: {}".format(e))
    # Vypíše: Nastala chyba: Pøistoupil jste k neexistujícímu prvku v seznamu.
except (TypeError, NameError):  # Více výjimek lze zachytit najednou
    pass  # Pass znamená nedìlej nic - nepøíliš vhodný zpùsob ošetøení chyb
else:  # Volitelný blok else musí být až za bloky except
    print("OK!")  # Vypíše OK! v pøípadì, že nenastala žádná výjimka
finally:  # Blok finally se spustí nakonec za všech okolností
    print("Uvolníme zdroje, uzavøeme soubory...")

# Místo try/finally lze použít with pro automatické uvolnìní zdrojù
with open("soubor.txt") as soubor:
    for radka in soubor:
        print(radka)
        
        
slovnik = {"jedna": 1, "dva": 2, "tøi": 3}
iterovatelny_objekt = slovnik.keys()
print(iterovatelny_objekt)  # => dict_keys(["jedna", "dva", "tøi"]). Toto je iterovatelný objekt.
# Mùžeme použít cyklus for na jeho projití
for klic in iterovatelny_objekt:
    print(klic)    # vypíše postupnì: jedna, dva, tøi
# Ale nelze pøistupovat k prvkùm pod jejich indexem
iterovatelny_objekt[1]  # Vyhodí TypeError
# Všechny položky iterovatelného objektu lze získat jako seznam pomocí list()
list(slovnik.keys())  # => ["jedna", "dva", "tøi"]
# Z iterovatelného objektu lze vytvoøit iterátor
iterator = iter(iterovatelny_objekt)
# Iterátor je objekt, který si pamatuje stav v rámci svého iterovatelného objektu
# Další hodnotu dostaneme voláním next()
next(iterator)  # => "jedna"
# Iterátor si udržuje svùj stav v mezi jednotlivými voláními next()
next(iterator)  # => "dva"
next(iterator)  # => "tøi"
# Jakmile interátor vrátí všechna svá data, vyhodí výjimku StopIteration
next(iterator)  # Vyhodí StopIteration


####################################################
## 4. Funkce
####################################################
# global x vs local x
# Klíèové slovo lambda vytvoøí anonymní funkci
(lambda parametr: parametr > 2)(3)
# Lze použít funkce map() a filter() z funkcionálního programování
map(lambda x: x + 10, [1, 2, 3])
# => <map object at 0x0123467> - iterovatelný objekt s obsahem: [11, 12, 13]
filter(lambda x: x > 5, [3, 4, 5, 6, 7])
# => <filter object at 0x0123467> - iterovatelný objekt s obsahem: [6, 7]

# S generátorovou notací lze dosáhnout podobných výsledkù, ale vrací seznam
{i: i + 10 for i in [1, 2, 3]}  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]   # => [6, 7]
# Generátorová notace funguje i pro slovníky
{x: x**2 for x in range(1, 5)}  # => {1: 1, 2: 4, 3: 9, 4: 16}
# A také pro množiny
{pismeno for pismeno in "abeceda"}  # => {'e', 'c', 'b', 'd', 'a'}


####################################################
## 5. Tøídy
####################################################

# Tøída Clovek je potomkem (dìdí od) tøídy object
class Clovek(object):

    # Atribut tøídy - je sdílený všemi instancemi
    druh = "H. sapiens"

    # Toto je kostruktor. Je volán, když vytváøíme instanci tøídy. Dvì
    # podtržítka na zaèátku a na konci znaèí, že se jedná o atribut nebo
    # objekt využívaný Pythonem ke speciálním úèelùm, ale mùžete sami
    # definovat jeho chování. Metody jako __init__, __str__, __repr__
    # a další se nazývají "magické metody". Nikdy nepoužívejte toto
    # speciální pojmenování pro bìžné metody.
    def __init__(self, jmeno):
        # Pøiøazení parametru do atributu instance jmeno
        self.jmeno = jmeno

    # Metoda instance - všechny metody instance mají "self" jako první parametr
    def rekni(self, hlaska):
        return "{jmeno}: {hlaska}".format(jmeno=self.jmeno, hlaska=hlaska)

    # Metoda tøídy - sdílená všemi instancemi
    # Dostává jako první parametr tøídu, na které je volána
    @classmethod
    def vrat_druh(cls):
        return cls.druh

    # Statická metoda je volána bez reference na tøídu nebo instanci
    @staticmethod
    def odkaslej_si():
        return "*ehm*"


# Vytvoøení instance
d = Clovek(jmeno="David")
a = Clovek("Adéla")
print(d.rekni("ahoj"))    # Vypíše: "David: ahoj"
print(a.rekni("nazdar"))  # Vypíše: "Adéla: nazdar"

# Volání tøídní metody
d.vrat_druh()  # => "H. sapiens"

# Zmìna atributu tøídy
Clovek.druh = "H. neanderthalensis"
d.vrat_druh()  # => "H. neanderthalensis"
a.vrat_druh()  # => "H. neanderthalensis"

# Volání statické metody
Clovek.odkaslej_si()  # => "*ehm*"


####################################################
## 6. Moduly
####################################################

# Lze importovat moduly
import math
print(math.sqrt(16.0))  # => 4.0

# Lze také importovat pouze vybrané funkce z modulu
from math import ceil, floor
print(ceil(3.7))   # => 4.0
print(floor(3.7))  # => 3.0

# Mùžete také importovat všechny funkce z modulu, ale radši to nedìlejte
from math import *

# Mùžete si pøejmenovat modul pøi jeho importu
import math as m
math.sqrt(16) == m.sqrt(16)  # => True

# Modul v Pythonu není nic jiného, než obyèejný soubor .py
# Mùžete si napsat vlastní a prostì ho importovat podle jména
from muj_modul import moje_funkce  # Nyní vyhodí ImportError - muj_modul neexistuje

# Funkcí dir() lze zjistit, co modul obsahuje
import math
dir(math)

####################################################
## 7. Pokroèilé
####################################################

# Generátory jsou funkce, které místo return obsahují yield
def nasobicka_2(sekvence):
    for i in sekvence:
        print("Zpracovávám èíslo {}".format(i))
        yield 2 * i

# Generátor generuje hodnoty postupnì, jak jsou potøeba. Místo toho, aby vrátil
# celou sekvenci s prvky vynásobenými dvìma, provádí jeden výpoèet v každé iteraci.
for nasobek in nasobicka_2([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    # Vypíše postupnì: "Zpracovávám èíslo 1",  ...,  "Zpracovávám èíslo 5"
    if nasobek >= 10:
        break

# Funkce range() je také generátor - vytváøení seznamu 900000000 prvkù by zabralo
# hodnì èasu i pamìti, proto se místo toho èísla generují postupnì.
for nasobek in nasobicka_2(range(900000000)):
    # Vypíše postupnì: "Zpracovávám èíslo 1",  ...,  "Zpracovávám èíslo 5"
    if nasobek >= 10:
        break


# Dekorátory jsou funkce, které se používají pro obalení jiné funkce, èímž mohou
# pøidávat nebo mìnit její stávající chování. Funkci dostávají jako parametr
# a typicky místo ní vrací jinou, která uvnitø volá tu pùvodní.

def nekolikrat(puvodni_funkce):
    def opakovaci_funkce(*args, **kwargs):
        for i in range(3):
            puvodni_funkce(*args, **kwargs)

    return opakovaci_funkce


@nekolikrat
def pozdrav(jmeno):
    print("Mìj se {}!".format(jmeno))

pozdrav("Pepo")  # Vypíše 3x: "Mìj se Pepo!"


# Data Analysis: Tabular Data ---------------------------------------------
# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

import pandas as pd
import numpy as np
import statistics
import datetime
import calendar

s = pd.Series([1, 3, 5, np.nan, 6, 8])

# pd.read_csv
# bool, str, int, float, pd.to_datetime


df = pd.DataFrame(
    np.random.randn(6, 4),
    index=pd.date_range("20130101", periods=6),
    columns=list("ABCD")
    )

df2 = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp("20130102"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo"
    })
df2["E"] = ["one", "two", "three", "three"]

df3 = pd.DataFrame({
    'a': ['a', 'b'] * 3,
    'b': [7,4,3,12,1,11],
    'c': [1,-1]*3
    })

df2.index
df2.columns
df2.to_numpy()
df2.describe()
df2.T # transpose

# select
df[["A", "B"]] # df["A"]
df.iloc[:,0:2]
df.iloc[:, [0, 2]]
df.loc[:, ["A", "B"]]

# arrange
df.sort_index(axis=0, ascending=False) # axis 0 znamenÃ¡ by row index, axis 1 by bylo by column index
df.sort_values(by=["A", "B"], ascending=[True, True])

# filter
filtering_input = 0
df.query('A > @filtering_input')
df.loc["20130102":"20130104", ]
df.iloc[3:5, ]
df.iloc[[1, 2, 4], ]
df[df["A"] > 0]
df2[df2["E"].isin(["two", "four"])]

# mutate
df.at["0", "A"] = 0.0
df.iat[0, 1] = 0
df.loc[:, "D"] = np.array([5] * len(df))
df.assign(
    assigned_column = lambda x: x.A + x.B + x.C,
    assigned_column2 = lambda x: x.C + x.D
    )
df.assign(assigned_column = lambda x: x.A.mean() + x.B)
df3.groupby('a').apply(lambda x: x.assign(assigned_apply_col = (x.b - x.b.mean())/statistics.stdev(x.b))).reset_index(drop=True)
df3.groupby('a').apply(lambda x: x.assign(pctg_col = x.b/x.b.sum())).reset_index(drop=True)

# summarise
df3.groupby('a', as_index=False)['b'].first()
df3.groupby('a', as_index=False)['b'].\
    agg([np.sum, np.mean, np.std])\
    .rename(columns={"sum": "foo", "mean": "bar", "std": "baz"})
df3.groupby('a', as_index=False)['b']\
    .agg([lambda x: x.max() - x.min(), lambda x: x.median() - x.mean()])\
    .rename(columns={"<lambda_0>": "first_lambda", "<lambda_1>": "second_lambda"})

# group_by & split
df3 = pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
df3.groupby(["X"]).get_group("A")
[v for k, v in df3.groupby('X')]
# pandas v 1.1 can override the default behaviour to drop na values

# joins
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
pd.merge(left, right, on="key")

# pivot_longer = Melt
df = pd.DataFrame({
    "first": ["John", "Mary"],
    "last": ["Doe", "Bo"],
    "height": [5.5, 6.0],
    "weight": [130, 150]
    })

df.melt(id_vars=["first", "last"]) # id_vars, value_vars (value_name, var_name)

df.melt(id_vars=None) # all vars

# pivot_wider pivot/pivot_table
df4 = pd.DataFrame({
    "first": ["John", "Mary", "Steve"]*2,
    "last": ["Doe"]*3 + ["Smith"]*3,
    "height": [5,4,3,7,8,12]
    })
df4 = df4.pivot_table(index="first",columns=["last"], values=["height"], aggfunc=np.sum)
df4.columns = ['_'.join(i) for i in df4.columns.values]
df4 = df4.reset_index()

# rowwise
for index, row in df.iterrows():
    print(row['A'], row['B'])
    print(df.loc[index,])


# datetime
dt = datetime.datetime.now()
datetime.date.today()
pd.to_datetime('2019-10-31 00:00:00')
dt.month
dt.year
dt.weekday() + 1 # number
calendar.day_name[dt.weekday()] # 'Monday'
rng = pd.date_range(
    start=datetime.date.today(),
    end=datetime.date.today()+datetime.timedelta(days=10),
    freq="D"
    )

pd.to_datetime(dt).to_period('M').to_timestamp() # floor_date
dt + pd.tseries.offsets.MonthEnd(0, normalize=True) # celing date; 0 works also for e.g. 2020-01-31
dt + datetime.timedelta(days=1) # add days

# df time series date_from, date_to
v_date_range = np.vectorize(pd.date_range)
df = pd.DataFrame({'dates_by_hrs':v_date_range(pd.to_datetime(["2020-01-01", "2020-01-02"]), pd.to_datetime(["2020-01-02", "2020-02-28"]), freq="H", tz="Europe/Prague")})
df = df.explode("dates_by_hrs")

############################################################
# Visualisation https://plotly.com/python/line-and-scatter/
############################################################
import plotly.express as px

# scatter plot
df = px.data.iris()
fig = px.scatter(
    df, x="sepal_width", y="sepal_length", color="species",
    marginal_y="violin", marginal_x="box",
    trendline="ols", template="simple_white"
)
fig.show()

# line plot
df = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(df, x='year', y='lifeExp', color='country')
fig.show()


############################################################
# Stat & ML https://www.w3schools.com/python/python_ml_multiple_regression.asp
############################################################
# linear regression
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

# polynomial regression
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# linear model
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv("https://www.w3schools.com/python/cars.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regr.predict([scaled[0]])


#############################################################################
# sparkline
#############################################################################

# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
# https://mode.com/example-gallery/python_dataframe_styling/

import numpy as np
import pandas as pd

def sparkline(data, figsize=(4, 0.25), **kwags):
    """
    Returns an HTML image tag containing a base64 encoded sparkline style plot
    """
    from matplotlib import pyplot as plt
    import base64
    from io import BytesIO
    
    data = list(data)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
    ax.plot(data)
    for k,v in ax.spines.items():
        v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])    

    plt.plot(len(data) - 1, data[len(data) - 1], 'r.')

    ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1)
    
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(base64.b64encode(img.read()).decode())

df = pd.DataFrame(
    np.random.randn(50, 4),
    columns=list("ABCD"))\
    .cumsum()\
    .melt()
    
html_df = df\
    .groupby("variable")\
    .agg(lambda x: sparkline(list(x)))\
    .style\
    .render()


with open('tst.html', 'w') as f:
    f.write(html_df)


#############################################################################
# html styling
#############################################################################

import pandas as pd
import seaborn as sns
import webbrowser

df = pd.DataFrame({
    "money_CZK": [987654321.123456, 0, -1000000000],
    "percentage": [0.99, 1.44, 0.56],
    "check": [True, True, False]
    })

def html_pos_neg_green_red(s):
    if s < 0:
        return 'background-color: #FFCCCB'
    elif s > 0:
        return 'background-color: lightgreen'


def html_false_or_neg_red(s):
    return ["background-color: #2b6e33" if v == True or v > 0 else "background-color: #C00000" for v in s]

def html_creation(df):
    caption_props = [
          ('font-size', '25px'),
          ('color', 'white'),
          ('background-color', '#595b5b')
      ]
    
    # Set CSS properties for th elements in dataframe
    th_props = [
      ('font-size', '18px'),
      ('text-align', 'center'),
      ('font-weight', 'normal'),
      ('color', 'white'),
      ('background-color', '#595b5b')
      ]
    
    # Set CSS properties for td elements in dataframe
    td_props = [
      ('font-size', '14px'),
      ('text-align', 'right'),
      ('border', '1px dotted grey') 
      ]
    
    # Set table styles
    styles = [
        dict(selector="caption", props=caption_props),
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)
      ]

    html = df\
        .style\
        .set_table_styles(styles)\
        .apply(lambda x: ["color: white" for v in x], subset=['check'])\
        .apply(html_false_or_neg_red, subset=['check'])\ # .applymap(html_pos_neg_green_red, subset=['check'])\
        .format("{:20,.0f} CZK", subset=['money_CZK'])\
        .format("{:.2%}", subset=['percentage'])\
        .bar(align='mid', color=['#FFCCCB', 'lightgreen'], subset=['money_CZK'])\
        .set_caption('This is a caption')\
        .background_gradient(sns.light_palette("black", as_cmap=True), vmin=0, vmax=1, subset=['percentage'])\
        .hide_index()\
        .render()
    
    return html

html = html_creation(df)

with open('export.html', 'w') as f:
    f.write(html)

webbrowser.open('export.html')

