# djangoBootParser
We developed djangoBootParser, a easy-to-use parsing tool to train your own parser in two click. It is developped especially to help building more quickly treebank of low-ressource language.<br>

To begin, we tested 5 recent parsers: 
* UDify (Kondratyuk and Straka 2019), 
* Hopsparser (Grobol and Crabbé 2021),
* Trankit (Nguyen et al. 2021), 
* Stanza (Qi et al 2020), 
* BertForDeprel (Guiller 2020)

on 5 on 5 typologically diverse languages, English (en), French (fr), Chinese (zh), Japanese (ja) and Arabic (ar) with data size 10, 30, 50, 100, 300, 500. Subsequently, we tested UDify and Trankit that outperformed others on all available languages with enough data in SUD2.10.<br>


This backend is integrated into arboratogrew and you can directly use it there for your project in [arboratorgrew](https://arboratorgrew.elizia.net/#/). Just set parameters such as the gold files as training set, epochs etc then click the 'begin parse button'. A polling will be launched to show current working step such as 'data preparation', 'training', 'parsing' etc.<br>


We suggest using UDify for dataset less than 100 sentences and Trankit otherwise.<br>

More detaile will be given soon in this readme.<br>

