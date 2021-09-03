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
## 1. Primitivn� datov� typy a oper�tory
####################################################

# A� na d�len�, kter� vrac� desetinn� ��slo
35 / 5  # => 7.0
# P�i celo��seln�m d�len� je desetinn� ��st o��znuta (pro kladn� i z�porn� ��sla)
5 // 3       # => 1
5.0 // 3.0   # => 1.0  #  celo��seln� d�lit lze i desetinn�m ��slem
-5 // 3      # => -2
-5.0 // 3.0  # => -2.0
# Pokud pou�ijete desetinn� ��slo, v�sledek je j�m tak�
3 * 2.0  # => 6.0
# Modulo
7 % 3  # => 1
# Mocn�n� (x na y-tou)
2**4  # => 16

# Pou��v�n� logick�ch oper�tor� s ��sly
0 and 2     # => 0
-5 or 0     # => -5

# P�i porovn�n� s boolean hodnotou nepou��vejte oper�tor rovnosti "==". 
# Stejn� jako u hodnoty None.
# Viz PEP8: https://www.python.org/dev/peps/pep-0008/ 
0 is False  # => True
2 is True   # => False
1 is True   # => True
"n�co" is None  # => False
None is None    # => True
# None, 0, a pr�zdn� �et�zec/seznam/N-tice/slovn�k/mno�ina se vyhodnot� jako False
# V�e ostatn� se vyhodnot� jako True
bool(0)        # => False
bool("")       # => False
bool([])       # => False
bool(tuple())  # => False
bool({})       # => False
bool(set())    # => False

# �et�zec lze pova�ovat za seznam znak�
"Toto je �et�zec"[0]  # => 'T'

# .format lze pou��t ke skl�d�n� �et�zc�
"{} mohou b�t {}".format("�et�zce", "skl�d�ny")
# Form�tovac� argumenty m��ete opakovat
"{0} {1} st��ka�ek st��kalo p�es {0} {1} st�ech".format("t�i sta t�icet t�i", "st��brn�ch")
# Pokud nechcete po��tat, m��ete pou��t pojmenovan� argumenty
"{jmeno} si dal {jidlo}".format(jmeno="Franta", jidlo="gul�")  # => "Franta si dal gul�"

####################################################
## 2. Prom�nn� a kolekce
####################################################
# Seznam (je kolekce) se pou��v� pro ukl�d�n� sekvenc�
sez = []
# Lze ho rovnou naplnit
jiny_seznam = [4, 5, 6]
# Na konec seznamu se p�id�v� pomoc� append
sez.append(1)    # sez je nyn� [1]
sez.append(2)    # sez je nyn� [1, 2]
sez.append(4)    # sez je nyn� [1, 2, 4]
sez.append(3)    # sez je nyn� [1, 2, 4, 3]
# Z konce se odeb�r� se pomoc� pop
sez.pop()        # => 3 a sez je nyn� [1, 2, 4]
# Vlo�me trojku zp�tky
sez.append(3)    # sez je nyn� znovu [1, 2, 4, 3]
# P��stup k prvk�m funguje jako v poli
sez[0]  # => 1
# M�nus po��t� odzadu (-1 je posledn� prvek)
sez[-1]  # => 3
# P��stup mimo seznam vyhod� IndexError
sez[4]  # Vyhod� IndexError
# Pomoc� �ez� lze ze seznamu vyb�rat r�zn� intervaly
# (pro matematiky: jedn� se o uzav�en�/otev�en� interval)
sez[1:3]  # => [2, 4]
# Od��znut� za��tku
sez[2:]  # => [4, 3]
# Od��znut� konce
sez[:3]  # => [1, 2, 4]
# Vybr�n� ka�d�ho druh�ho prvku
sez[::2]  # =>[1, 4]
# Vr�cen� seznamu v opa�n�m po�ad�
sez[::-1]  # => [3, 4, 2, 1]
# Lze pou��t jakoukoliv kombinaci parametr� pro vytvo�en� slo�it�j��ho �ezu
# sez[zacatek:konec:krok]
# Odeb�rat prvky ze seznamu lze pomoc� del
del sez[2]   # sez je nyn� [1, 2, 3]
# Seznamy m��ete slu�ovat
# Hodnoty sez a jiny_seznam p�itom nejsou zm�n�ny
sez + jiny_seznam   # => [1, 2, 3, 4, 5, 6]
# Spojit seznamy lze pomoc� extend
sez.extend(jiny_seznam)   # sez je nyn� [1, 2, 3, 4, 5, 6]
# Kontrola, jestli prvek v seznamu existuje, se prov�d� pomoc� in
1 in sez  # => True
# D�lku seznamu lze zjistit pomoc� len
len(sez)  # => 6


# N-tice je jako seznam, ale je nem�nn�
ntice = (1, 2, 3)
ntice[0]      # => 1
ntice[0] = 3  # Vyhod� TypeError
# S n-ticemi lze d�lat v�t�inu operac�, jako se seznamy
len(ntice)         # => 3
ntice + (4, 5, 6)  # => (1, 2, 3, 4, 5, 6)
ntice[:2]          # => (1, 2)
2 in ntice         # => True
# N-tice (nebo seznamy) lze rozbalit do prom�nn�ch jedn�m p�i�azen�m
a, b, c = (1, 2, 3)  # a je nyn� 1, b je nyn� 2 a c je nyn� 3
# N-tice jsou vytv��eny automaticky, kdy� vynech�te z�vorky
d, e, f = 4, 5, 6
# Prohozen� prom�nn�ch je tak velmi snadn�
e, d = d, e  # d je nyn� 5, e je nyn� 4


# Slovn�ky ukl�daj� kl��e a hodnoty
prazdny_slovnik = {}
# Lze je tak� rovnou naplnit
slovnik = {"jedna": 1, "dva": 2, "t�i": 3}
# P�istupovat k hodnot�m lze pomoc� []
slovnik["jedna"]  # => 1
# V�echny kl��e dostaneme pomoc� keys() jako iterovateln� objekt. Nyn� je�t�
# pot�ebujeme obalit vol�n� v list(), abychom dostali seznam. To rozebereme
# pozd�ji. Pozor, �e jak�koliv po�ad� kl��� nen� garantov�no - m��e b�t r�zn�.
list(slovnik.keys())  # => ["dva", "jedna", "t�i"]
# V�echny hodnoty op�t jako iterovateln� objekt z�sk�me pomoc� values(). Op�t
# tedy pot�ebujeme pou��t list(), abychom dostali seznam. Stejn� jako
# v p�edchoz�m p��pad�, po�ad� nen� garantov�no a m��e b�t r�zn�
list(slovnik.values())  # => [1, 2, 3]
# Oper�torem in se lze dot�zat na p��tomnost kl��e
"jedna" in slovnik  # => True
1 in slovnik        # => False
# P��stup k neexistuj�c�mu kl��i vyhod� KeyError
slovnik["�ty�i"]  # Vyhod� KeyError
# Metoda get() funguje podobn� jako [], ale vr�t� None m�sto vyhozen� KeyError
slovnik.get("jedna")   # => 1
slovnik.get("�ty�i")   # => None
# Metod� get() lze p�edat i v�choz� hodnotu m�sto None
slovnik.get("jedna", 4)   # => 1
slovnik.get("�ty�i", 4)   # => 4
# metoda setdefault() vlo�� prvek do slovn�ku pouze pokud tam takov� kl�� nen�
slovnik.setdefault("p�t", 5)  # slovnik["p�t"] je nastaven na 5
slovnik.setdefault("p�t", 6)  # slovnik["p�t"] je po��d 5
# P�id�n� nov� hodnoty do slovn�ku
slovnik["�ty�i"] = 4
# Hromadn� aktualizovat nebo p�idat data lze pomoc� update(), parametrem je op�t slovn�k
slovnik.update({"�ty�i": 4})  # slovnik je nyn� {"jedna": 1, "dva": 2, "t�i": 3, "�ty�i": 4, "p�t": 5}
# Odeb�rat ze slovn�ku dle kl��e lze pomoc� del
del slovnik["jedna"]  # odebere kl�� "jedna" ze slovnik


# Mno�iny ukl�daj� ... p�ekvapiv� mno�iny
prazdna_mnozina = set()
# Tak� je lze rovnou naplnit. A ano, budou se v�m pl�st se slovn�ky. Bohu�el.
mnozina = {1, 1, 2, 2, 3, 4}  # mnozina je nyn� {1, 2, 3, 4}
# P�id�n� polo�ky do mno�iny
mnozina.add(5)  # mnozina je nyn� {1, 2, 3, 4, 5}
# Pr�nik lze ud�lat pomoc� oper�toru &
jina_mnozina = {3, 4, 5, 6}
mnozina & jina_mnozina  # => {3, 4, 5}
# Sjednocen� pomoc� oper�toru |
mnozina | jina_mnozina  # => {1, 2, 3, 4, 5, 6}
# Rozd�l pomoc� oper�toru -
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}
# Oper�torem in se lze dot�zat na p��tomnost prvku v mno�in�
2 in mnozina  # => True
9 in mnozina  # => False


####################################################
## 3. ��zen� toku programu, cykly
####################################################

x = 0
while x < 4:
    print(x)
    x += 1  # Zkr�cen� z�pis x = x + 1. Pozor, ��dn� x++ neexisuje.
    

# V�jimky lze o�et�it pomoc� bloku try/except(/else/finally)
try:
    # Pro vyhozen� v�jimky pou�ijte raise
    raise IndexError("P�istoupil jste k neexistuj�c�mu prvku v seznamu.")
except IndexError as e:
    print("Nastala chyba: {}".format(e))
    # Vyp�e: Nastala chyba: P�istoupil jste k neexistuj�c�mu prvku v seznamu.
except (TypeError, NameError):  # V�ce v�jimek lze zachytit najednou
    pass  # Pass znamen� ned�lej nic - nep��li� vhodn� zp�sob o�et�en� chyb
else:  # Voliteln� blok else mus� b�t a� za bloky except
    print("OK!")  # Vyp�e OK! v p��pad�, �e nenastala ��dn� v�jimka
finally:  # Blok finally se spust� nakonec za v�ech okolnost�
    print("Uvoln�me zdroje, uzav�eme soubory...")

# M�sto try/finally lze pou��t with pro automatick� uvoln�n� zdroj�
with open("soubor.txt") as soubor:
    for radka in soubor:
        print(radka)
        
        
slovnik = {"jedna": 1, "dva": 2, "t�i": 3}
iterovatelny_objekt = slovnik.keys()
print(iterovatelny_objekt)  # => dict_keys(["jedna", "dva", "t�i"]). Toto je iterovateln� objekt.
# M��eme pou��t cyklus for na jeho projit�
for klic in iterovatelny_objekt:
    print(klic)    # vyp�e postupn�: jedna, dva, t�i
# Ale nelze p�istupovat k prvk�m pod jejich indexem
iterovatelny_objekt[1]  # Vyhod� TypeError
# V�echny polo�ky iterovateln�ho objektu lze z�skat jako seznam pomoc� list()
list(slovnik.keys())  # => ["jedna", "dva", "t�i"]
# Z iterovateln�ho objektu lze vytvo�it iter�tor
iterator = iter(iterovatelny_objekt)
# Iter�tor je objekt, kter� si pamatuje stav v r�mci sv�ho iterovateln�ho objektu
# Dal�� hodnotu dostaneme vol�n�m next()
next(iterator)  # => "jedna"
# Iter�tor si udr�uje sv�j stav v mezi jednotliv�mi vol�n�mi next()
next(iterator)  # => "dva"
next(iterator)  # => "t�i"
# Jakmile inter�tor vr�t� v�echna sv� data, vyhod� v�jimku StopIteration
next(iterator)  # Vyhod� StopIteration


####################################################
## 4. Funkce
####################################################
# global x vs local x
# Kl��ov� slovo lambda vytvo�� anonymn� funkci
(lambda parametr: parametr > 2)(3)
# Lze pou��t funkce map() a filter() z funkcion�ln�ho programov�n�
map(lambda x: x + 10, [1, 2, 3])
# => <map object at 0x0123467> - iterovateln� objekt s obsahem: [11, 12, 13]
filter(lambda x: x > 5, [3, 4, 5, 6, 7])
# => <filter object at 0x0123467> - iterovateln� objekt s obsahem: [6, 7]

# S gener�torovou notac� lze dos�hnout podobn�ch v�sledk�, ale vrac� seznam
{i: i + 10 for i in [1, 2, 3]}  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]   # => [6, 7]
# Gener�torov� notace funguje i pro slovn�ky
{x: x**2 for x in range(1, 5)}  # => {1: 1, 2: 4, 3: 9, 4: 16}
# A tak� pro mno�iny
{pismeno for pismeno in "abeceda"}  # => {'e', 'c', 'b', 'd', 'a'}


####################################################
## 5. T��dy
####################################################

# T��da Clovek je potomkem (d�d� od) t��dy object
class Clovek(object):

    # Atribut t��dy - je sd�len� v�emi instancemi
    druh = "H. sapiens"

    # Toto je kostruktor. Je vol�n, kdy� vytv���me instanci t��dy. Dv�
    # podtr��tka na za��tku a na konci zna��, �e se jedn� o atribut nebo
    # objekt vyu��van� Pythonem ke speci�ln�m ��el�m, ale m��ete sami
    # definovat jeho chov�n�. Metody jako __init__, __str__, __repr__
    # a dal�� se naz�vaj� "magick� metody". Nikdy nepou��vejte toto
    # speci�ln� pojmenov�n� pro b�n� metody.
    def __init__(self, jmeno):
        # P�i�azen� parametru do atributu instance jmeno
        self.jmeno = jmeno

    # Metoda instance - v�echny metody instance maj� "self" jako prvn� parametr
    def rekni(self, hlaska):
        return "{jmeno}: {hlaska}".format(jmeno=self.jmeno, hlaska=hlaska)

    # Metoda t��dy - sd�len� v�emi instancemi
    # Dost�v� jako prvn� parametr t��du, na kter� je vol�na
    @classmethod
    def vrat_druh(cls):
        return cls.druh

    # Statick� metoda je vol�na bez reference na t��du nebo instanci
    @staticmethod
    def odkaslej_si():
        return "*ehm*"


# Vytvo�en� instance
d = Clovek(jmeno="David")
a = Clovek("Ad�la")
print(d.rekni("ahoj"))    # Vyp�e: "David: ahoj"
print(a.rekni("nazdar"))  # Vyp�e: "Ad�la: nazdar"

# Vol�n� t��dn� metody
d.vrat_druh()  # => "H. sapiens"

# Zm�na atributu t��dy
Clovek.druh = "H. neanderthalensis"
d.vrat_druh()  # => "H. neanderthalensis"
a.vrat_druh()  # => "H. neanderthalensis"

# Vol�n� statick� metody
Clovek.odkaslej_si()  # => "*ehm*"


####################################################
## 6. Moduly
####################################################

# Lze importovat moduly
import math
print(math.sqrt(16.0))  # => 4.0

# Lze tak� importovat pouze vybran� funkce z modulu
from math import ceil, floor
print(ceil(3.7))   # => 4.0
print(floor(3.7))  # => 3.0

# M��ete tak� importovat v�echny funkce z modulu, ale rad�i to ned�lejte
from math import *

# M��ete si p�ejmenovat modul p�i jeho importu
import math as m
math.sqrt(16) == m.sqrt(16)  # => True

# Modul v Pythonu nen� nic jin�ho, ne� oby�ejn� soubor .py
# M��ete si napsat vlastn� a prost� ho importovat podle jm�na
from muj_modul import moje_funkce  # Nyn� vyhod� ImportError - muj_modul neexistuje

# Funkc� dir() lze zjistit, co modul obsahuje
import math
dir(math)

####################################################
## 7. Pokro�il�
####################################################

# Gener�tory jsou funkce, kter� m�sto return obsahuj� yield
def nasobicka_2(sekvence):
    for i in sekvence:
        print("Zpracov�v�m ��slo {}".format(i))
        yield 2 * i

# Gener�tor generuje hodnoty postupn�, jak jsou pot�eba. M�sto toho, aby vr�til
# celou sekvenci s prvky vyn�soben�mi dv�ma, prov�d� jeden v�po�et v ka�d� iteraci.
for nasobek in nasobicka_2([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    # Vyp�e postupn�: "Zpracov�v�m ��slo 1",  ...,  "Zpracov�v�m ��slo 5"
    if nasobek >= 10:
        break

# Funkce range() je tak� gener�tor - vytv��en� seznamu 900000000 prvk� by zabralo
# hodn� �asu i pam�ti, proto se m�sto toho ��sla generuj� postupn�.
for nasobek in nasobicka_2(range(900000000)):
    # Vyp�e postupn�: "Zpracov�v�m ��slo 1",  ...,  "Zpracov�v�m ��slo 5"
    if nasobek >= 10:
        break


# Dekor�tory jsou funkce, kter� se pou��vaj� pro obalen� jin� funkce, ��m� mohou
# p�id�vat nebo m�nit jej� st�vaj�c� chov�n�. Funkci dost�vaj� jako parametr
# a typicky m�sto n� vrac� jinou, kter� uvnit� vol� tu p�vodn�.

def nekolikrat(puvodni_funkce):
    def opakovaci_funkce(*args, **kwargs):
        for i in range(3):
            puvodni_funkce(*args, **kwargs)

    return opakovaci_funkce


@nekolikrat
def pozdrav(jmeno):
    print("M�j se {}!".format(jmeno))

pozdrav("Pepo")  # Vyp�e 3x: "M�j se Pepo!"


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
df.sort_index(axis=0, ascending=False) # axis 0 znamená by row index, axis 1 by bylo by column index
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

