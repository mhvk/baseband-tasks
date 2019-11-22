import re
import os
from html.parser import HTMLParser

from astropy.io import fits
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename


psrfits_doc_path = get_pkg_data_filename('../../data/PsrfitsDocumentation.html')
fits_table_map = {'BINTABLE': fits.BinTableHDU,
                  'PRIMARY': fits.PrimaryHDU}
column_map = {'TTYPE': 'name', 'TFORM': 'format', 'TUNIT': 'unit',
              'TDIM': 'dim'}
unit_rex = re.compile(r'\[(.+)\]')
dim_rex = re.compile(r'N[A-Z]+')
tdim_rex = re.compile(r'\((.*?)\)')


class MyHTMLParser(HTMLParser):
    section = False
    collect = False
    add_nl = False
    extensions = {}
    extname = ''
    data = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'p' or (tag == 'div' and attrs == [('class', 'indent')]):
            self.collect = True
            self.data = ''
        elif tag == 'br':
            self.add_nl = True
        elif tag == 'h3':
            if self.section:
                self.dump()
            self.section = True
            self.headers = []

    def handle_endtag(self, tag):
        if tag not in ('p', 'div') or not self.collect:
            return

        self.collect = False
        data = self.data
        if tag == 'p' and (
                data.startswith('COMMENT') or
                data.startswith('END') or
                (len(data) > 8 and data[8] == '=')):
            if data.startswith('EXTNAME'):
                self.extname = data.split(' ')[2]
            self.headers += data.split('\n')
        elif tag == "div" and self.headers and (
                not data.startswith('Standard')):
            # description for last header.
            self.headers[-1] = self.headers[-1] + '\n# ' + data

    def handle_data(self, data):
        data = data.replace('\n', '')
        if self.collect:
            self.data += ('\n' if self.add_nl else '') + data
        self.add_nl = False

    def dump(self):
        self.extensions[self.extname] = '\n'.join(self.headers)


def ext2hdu(extension):
    """Turn extension information from htm file into a FITS HDU.

    The extension will have a header and columns set as appropriate.
    The original cards, including possible descriptions are stored
    as hdu.header_cards and hdu.column_cards, respectively.
    """
    header_cards = []
    column_cards = []
    # parse extension lines
    lines = extension.splitlines()
    icol = 0
    card = None

    for line in lines:
        line = line.strip()
        if line == "" or line == "END":
            continue

        if line.startswith("#"):
            # Description for previous card.
            assert not hasattr(card, 'description')
            card.description = line[1:].strip()
            continue

        is_col = '#' in line[1:8]
        if is_col:
            if line[0:5] == 'TTYPE':
                icol += 1
                colinfo = []
                column_cards.append(colinfo)
            # This is a column.  Put in the number
            line = line.replace('#', ' ')

        if (is_col or line.startswith('XTENSION') or
                line.startswith('EXTNAME')):
            # Value should be a string but isn't in the .html file.
            line = line[:10] + "'" + line[10:27] + "'" + line[29:]
        elif "* /" in line:
            # A plain * cannot be parsed; it needs to be a string.
            line = line.replace("=                    * /",
                                "= '*       '           /")

        # A few lines are too long; remove some excess space from those.
        if len(line) > 80:
            pre, _, post = line.rpartition(' '*(len(line)-80))
            line = pre + post

        card = fits.Card.fromstring(line.strip())

        if card.comment.startswith('['):
            m = re.search(unit_rex, card.comment).groups()[0]
            try:
                card.unit = u.Unit(m)
            except Exception:
                pass

        if is_col:
            colinfo.append(card)
        else:
            header_cards.append(card)
            if card.keyword == 'SIMPLE':
                hdu_cls = fits.PrimaryHDU
            elif card.keyword == 'XTENSION':
                # NOTE, this should be the right way to do it,
                # but right now it only handles one case, BINTABLE.
                hdu_cls = fits_table_map[card.value]

    columns = []
    for colinfo in column_cards:
        # Collate column cards.
        kwargs = {fits.column.KEYWORD_TO_ATTRIBUTE[card.keyword]: card.value
                  for card in colinfo}
        if kwargs['format'] == 'V':  # not in FITS standard...
            kwargs['format'] = 'J'
        column = fits.Column(**kwargs)
        column.cards = colinfo
        columns.append(column)

    header = fits.Header(header_cards)
    if 'TFIELDS' in header and header['TFIELDS'] == '*':
        header['TFIELDS'] = 0

    hdu = hdu_cls(header=header)
    hdu.header_cards = header_cards
    if columns:
        hdu.columns = fits.ColDefs(columns)

    return hdu


def get_dtype(col_entry):
    """ Get the dimension from the column format field.
    """
    # Get dim from the format value
    # TODO, this split alway has an empty array
    dim = None
    dtype = None
    if col_entry['dim'] is not None:
        dim_strs = re.findall(tdim_rex, col_entry['dim'][1])
        # there could be options for dim
        dim = []
        for dim_str in dim_strs:
            dim.append(tuple(dim_str.split(',')))
    # parse data type
    format_fields = re.split(r'(\d+)', col_entry['format'][0])
    if len(format_fields) == 3:
        format_fields.remove("")
    if len(format_fields) == 2:
        # If the TDIM does not exist
        if dim is None:
            dim = [(format_fields[0], ), ]
        dtype = format_fields[1]
    elif len(format_fields) == 1:
        if dim is None:
            # parse dim from the description
            dim = [tuple(re.findall(dim_rex, col_entry['format'][1])), ]
        dtype = format_fields[0]
    return dim, dtype


parser = MyHTMLParser()
with open(psrfits_doc_path)as f:
    parser.feed(f.read())


hdu_templates = {}
for k, v in parser.extensions.items():
    if k == "":
        k = "PRIMARY"
    hdu_templates[k] = ext2hdu(v)
