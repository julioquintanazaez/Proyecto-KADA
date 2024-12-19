import re
import string
import nltk
from nltk.corpus import stopwords



nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.my_stopwords = ['granos', 'grano', 'largo', 'largos', 'criollo', 'negro', 'blanco', 'negros', 'blancos',
                              'importado', 'unidades', 'orgánico', 'molido', 'espresso', 'blanca',
                             'importada', 'usa', 'dale', 'llave', 'tradicional', 'clásico', 'refinada', 'ruano',
                             'tailandés', 'puro', 'aroma', 'cristal', 'instantáneo', 'original', 'natural',
                             'granulada', 'energy', 'soluble', 'nescafé', 'goya', 'fuerte', 'premium', 'sabor',
                              'cubita', 'yeya', 'mezcla', 'lata', 'boom', 'colacafé', 'cartones',
                             'turco', 'nestlé', 'morena', 'alessandro', 'rojo', 'gourmet', 'mujer','italiano',
                             'bolsa', 'ricuras', 'campo','refinado','sol','soya','vegetal','girasol','kada','percasol',
                             'oliva','cocina','limpro','goya','concentrado','soja','suavit','kent','boringer',
                             'didi','brasileño','villa','rica','extra','dualis','scotti','especial','super','yu','fina',
                             'vima','foods','levante','fulgor','royal','delicci','pullman','molde','perro','corona','ajo',
                             'lasqueado','empaquetado','bom','trenzado','albahaca','hamburguesa','miga','dorada','pequeño',
                             'bolita','cuerno','ajonjolí','bocaditos','baguette','redondo','tipo','suave','telera','integral',
                             'bonete','esponjoso','celimar','cebollas','semillas','viena','rebanado','cantolla','pakopan',
                             'bocadito','barra','rustiguet','wapa','santa','rita','onena','paquete','pureza','alta','refisal',
                             'común','anita','mccormick','yodada','marina','bueno','fina','pimienta','great','value','polasal',
                             'jabiana','catarinos','rosa','himalaya','marcum','apio','cebolla','selecto','catarinos',
                             'cour','di','mare','precíssimo','alimerka','margarita','master of mixes','gruesa','best','yet',
                             'refina','mais','doce','decancio','cancio','producida','eeuu','badia','ajo','mini','perca','sem',
                             'magnasur','ovilo','virgen'

                             ]

    def text_process(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\b(kilo|gramos|kg|g|ml|lb|kg|lt|l|ml|oz|gr)\b', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('spanish'))
        stop_words_p = stop_words.union(self.my_stopwords)
        text = ' '.join(word for word in text.split() if word not in stop_words_p)
        return text

    def text_process_recomender(self,text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\b(kilo|gramos|kg|g|ml|lb|kg|lt|l|ml|oz|gr)\b', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('spanish'))
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text


