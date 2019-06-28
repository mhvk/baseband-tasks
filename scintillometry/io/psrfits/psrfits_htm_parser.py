import re
import os
from html.parser import HTMLParser

from astropy.io import fits
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename


psrfits_doc_path = get_pkg_data_filename('../../data/PsrfitsDocumentation.html')
fits_table_map = {'BINTABLE': fits.BinTableHDU,
                  'PRIMARY': fits.PrimaryHDU}
unit_rex = re.compile(r'\[(.+)\]')


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


def parse_line(line):
    """parse_line indentifies the type of line and breaks lines to information
    segments.

    Retrun
    ------
    type of line, key word, value of the key, and comment about the key.
    """
    line_type = None
    key = ""
    value = ""
    comment = ""
    unit = None
    if line.strip() == "":
        line_type = "empty"
    elif line.startswith("#"):
        line_type = "description"
        value = line.replace("#", '').strip()
    elif line.startswith("COMMENT"):
        line_type = "comment"
        key = "COMMENT"
        value = line.replace("COMMENT", "")
    else:
        l_fields = re.split('= |/ ', line)
        if len(l_fields) <= 1:
            line_type = "empty"
        key = l_fields[0].strip()
        if key in ['TTYPE#', 'TFORM#', 'TUNIT#', 'TDIM#']:
            line_type = "column"
            key = key.replace("#", "")
            value = l_fields[1].strip()
        else:
            line_type = "card"
            value = l_fields[1].strip().replace("'", "")
        if len(l_fields) > 2:
            comment = l_fields[2].replace('#', '').strip()
            if comment.startswith('['):
                m = re.search(unit_rex, comment).groups()[0]
                try:
                    unit = u.Unit(m)
                except Exception:
                    pass

    return line_type, key, value, comment, unit


def ext2hdu(extension):
    hdu_parts = {'card': [], 'column': [], 'comment': []}
    hdu = None
    column_map = {'TTYPE': 'name', 'TFORM': 'format', 'TUNIT': 'unit',
                  'TDIM': 'dim'}
    cur_entry = {}
    last_entry = cur_entry
    # parse extension lines
    lines = extension.splitlines()
    ii = 0

    while lines[ii] != "END":
        line_type, key, value, comment, unit = parse_line(lines[ii])
        cur_type = cur_entry.get('type', '')
        if line_type == "empty":
            pass
        elif line_type == 'description':
            if cur_entry != {}:
                cur_entry['description'] += value
        elif line_type == 'comment':
            if cur_type == 'comment':
                cur_entry['value'] += value
            else:
                # record the current entry and init new entry
                hdu_parts[cur_entry['type']].append(cur_entry)
                last_entry = cur_entry
                cur_entry = {'name': key, 'value': value, 'type': line_type,
                             'after': last_entry['name'], 'description': ''}
        elif line_type == 'column':
            entry_key = column_map.get(key, None)
            if cur_type == 'column' and key != 'TTYPE':
                if entry_key is not None:
                    cur_entry[entry_key] = (value, comment)
            else:
                # Record the last entry and open a new entry for column
                hdu_parts[cur_entry['type']].append(cur_entry)
                last_entry = cur_entry
                cur_entry = {'name': None, 'format': None, 'unit': None,
                             'dim': None, 'description': '', 'type': line_type}
                if entry_key is not None:
                    cur_entry[entry_key] = (value, comment)

        elif line_type == 'card':
            if cur_entry != {}:
                hdu_parts[cur_entry['type']].append(cur_entry)
            last_entry = cur_entry
            cur_entry = {'name': key, 'value': value, 'unit': unit,
                         'comment': comment, 'description': '',
                         'type': line_type}
            # Search hdu type
            # When certain key words have been detected, it will call the
            # astropy fits hdu
            if key == 'SIMPLE':
                hdu = fits.PrimaryHDU
            elif key == 'XTENSION':
                # NOTE, this should be the right way to do it, but right now it
                # only handles one case, BINTABLE.
                hdu = fits_table_map[value]
            else:
                pass
        else:
            raise ValueError("Can not parse line '{}'".format(lines[ii]))
        ii += 1
    return (hdu, hdu_parts)


parser = MyHTMLParser()
with open(psrfits_doc_path)as f:
    parser.feed(f.read())
hdu_templates = {}
for k, v in parser.extensions.items():
    if k == "":
        k = "PRIMARY"
    hdu_templates[k] = ext2hdu(v)
