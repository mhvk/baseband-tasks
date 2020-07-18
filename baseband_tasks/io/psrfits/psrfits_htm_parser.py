# Licensed under the GPLv3 - see LICENSE
"""Parser of the PSRFITS definition file.

Parses the PSRFITS definition file from
https://www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html
(which is also available in ``baseband_tasks.data``)
to get template Header Data Units (HDUs) for the various extensions.

This is a two-step process, with the .htm file first parsed into
individual lines, which are either approximate `~astropy.io.fits.Card`
entries or descriptions of those cards, and then those lines
interpreted as actual `~astropy.io.fits.Card` instances, and combined
into headers, columns, and `~astropy.io.fits.PrimaryHDU` and
`~astropy.io.fits.BinTableHDU` instances.
"""
from html.parser import HTMLParser

from astropy.io import fits
from astropy.units import Unit

from baseband_tasks.data import PSRFITS_DOCUMENTATION


class MyHTMLParser(HTMLParser):
    """Parser for the PSR FITS definition.

    This follows the structure in the .html to create a number of
    ``extension`` entries which contain lines that define them.
    Those lines are close to the required format to define FITS Cards.
    """
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
                data.startswith('COMMENT')
                or data.startswith('END')
                or (len(data) > 8 and data[8] == '=')):
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


class PSRFITSCard(fits.Card):
    # Override __repr__ to show description if present.
    description = None

    def __repr__(self):
        out = super().__repr__()
        return out + '\n# ' + self.description if self.description else out


def ext2hdu(extension):
    """Turn extension information from htm file into a FITS HDU.

    The extension will have a header and columns set as appropriate.
    The original cards, including possible descriptions are stored
    as hdu.header_cards and hdu.column_cards, respectively.
    """
    cards = []
    # parse extension lines
    lines = extension.splitlines()
    icol = 0
    card_for_description = None

    for line in lines:
        line = line.strip()
        if line == "" or line == "END":
            continue

        if line.startswith("#"):
            # Description for previous card (or TTYPE card for column).
            assert card_for_description.description is None
            card_for_description.description = line[1:].strip()
            continue
            # For reference: this worked but gives very long FITS headers.
            # parts = textwrap.wrap(line[1:].strip(), width=70)
            # cards.extend([fits.Card('COMMENT', part) for part in parts])

        is_col = '#' in line[1:8]
        if is_col:
            if line[0:5] == 'TTYPE':
                icol += 1
            # Replace the placeholder with an actual number.
            line = line.replace('# ', '{:<2d}'.format(icol))

        if is_col or line.startswith('XTENSION') or line.startswith('EXTNAME'):
            # Value should be a string but quotes are omitted in the .htm file.
            line = line[:10] + "'" + line[10:27] + "'" + line[29:]
        elif "* /" in line:
            # A plain * cannot be parsed; it needs to be a string.
            line = line.replace("=                    * /",
                                "= '*       '           /")

        # A few lines are too long; remove some excess space from those,
        # by removing the sufficiently long set of spaces closest to the end.
        if len(line) > 80:
            pre, _, post = line.rpartition(' '*(len(line)-80))
            line = pre + post

        # Create a FITS Card using our description handling subclass.
        card = PSRFITSCard.fromstring(line)

        if card.comment.startswith('['):
            # Unit in the comments: store it for possible use.
            m = card.comment[1:].split(']')[0]
            if m not in ('v/c', 'MJD'):
                # TODO: support v/c as a special unit.
                card.unit = Unit(m)

        if card.keyword.startswith('TUNIT'):
            # Some units are in upper case, which won't get parsed correctly.
            if card.value in ('CM-3 PC', 'RAD M-2'):
                card.value = card.value.lower()
            # Sanity check.
            Unit(card.value)

        cards.append(card)
        if not is_col or line.startswith('TTYPE'):
            card_for_description = card

    header = fits.Header(cards)

    if 'SIMPLE' in header:
        assert icol == 0, 'non-table extension with columns!'
        hdu = fits.PrimaryHDU(header=header)
    else:
        # The number of columns is sometimes not defined or incorrect,
        # so just set it to the right number.
        assert icol > 0, 'table extension without columns!'
        header['TFIELDS'] = icol
        hdu = fits.BinTableHDU(data=fits.DELAYED, header=header)

    return hdu


parser = MyHTMLParser()
with open(PSRFITS_DOCUMENTATION)as f:
    parser.feed(f.read())


HDU_TEMPLATES = dict((k or "PRIMARY", ext2hdu(v))
                     for k, v in parser.extensions.items())
"""PSRFITS template HDUs.

With headers and column information as appropriate.
The cards used to make the HDU are available via the
``.header_cards`` and ``.column_cards`` attributes.
"""


# Free memory used by parsed file contents.
del parser
