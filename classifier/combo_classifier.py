import re

class ComboClassifier:
    def combo_clf(self, row):
        name = row['name'].lower()
        combo_name_query = re.compile(r".*combo.*", re.IGNORECASE)

        x_patterns_queries = [
            re.compile(r".*[0-9]+\s*[xX]\s*[0-9]+.*"),
            re.compile(r"\([0-9]+\s*[xX]\s*[0-9]+\s*[A-Za-z]+\)"),
            re.compile(r"[0-9]+\s*[xX]\s*[0-9]+\s+[A-Za-z]+"),
        ]
        plus_pack_queries = [
            re.compile(r".*\s+y\s+.*"),
            re.compile(r".*\+\s*.*"),
            re.compile(r".*\s*\+\s*.*"),
            re.compile(r".*pack.*"),
            re.compile(r".*pack\+.*"),
            re.compile(r".*\+pack.*"),
            re.compile(r".*pack\s*\+.*"),
            re.compile(r".*\+\s*pack.*"),
            re.compile(r".*pack\s*\*\s*\+.*"),
            re.compile(r".*\+\s*\*pack.*"),
            #re.compile(r"\(\d+\s*[×x]\s*\d+\s*[Uu]\)"),
            #re.compile(r"\(\d+\s*[uU]nidades\s*x\s*\d+\s*[gG]{1,2}\)"),
            #re.compile(r"\(\d+\s*[xX]\s*\d+\s*[gG]{1,2}\s*/\s*\d+\s*[oO]{1,2}z\)"),
           # re.compile(r"\(\d+\s*[uU]nidades\s*[xX]\s*\d+\s*[gG]{1,2}\)"),
           # re.compile(r'\(\d+\s+unidades\s+x\s+\d+\s+gr\)'),
            #re.compile(r'\(\d+\s+unidades\s+x\s+\d+\s+(ml|lt)\)'),
           # re.compile(r'(\w+)\s*y\s*(\w+)'),
           # re.compile(r'(\w+\s*\(\d+\w*\))\s*y\s*(\w+\s*\(\d+\w*\))')

        ]


        for exp in plus_pack_queries:
            if exp.search(name):
                return 'otros'
        return row['tag']