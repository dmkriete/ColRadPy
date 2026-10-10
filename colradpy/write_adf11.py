import numpy as np
import itertools
from datetime import date
from colradpy.colradpy_utility import element_name


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def _write_float_block(f, values, num=8):
    """Fixed-width F10.5 fields, `num` per line, no separator."""
    for chunk in grouper(num, values):
        f.write(''.join('{:10.5f}'.format(v) for v in chunk))
        f.write('\n')


def _write_int_block(f, values, num=8):
    """Fixed-width I5 fields, `num` per line."""
    for chunk in grouper(num, values):
        f.write(''.join('{:5d}'.format(v) for v in chunk))
        f.write('\n')


def _block_header(cls_field, iprt, igrd, z1, date_str):
    """Per-charge-state (or per-metastable-transition) block header.
    IPRT/IGRD are literal fields used by both formats: for unresolved files
    they are always 1; for resolved files IPRT indexes the parent metastable
    and IGRD indexes the child metastable.
    """
    return (' ' + '-' * 20 +
            '/ IPRT={:2d}  / IGRD={:2d}  /{:<8s}/ Z1={:2d}   / DATE= {}\n'
            .format(iprt, igrd, cls_field, z1, date_str))


def write_adf11(
    file_name, rates, nuclear_charge, izmin, izmax, dens, temp, metas=None,
    rate_type='SCD', reference='', date_str=None, comment='',
):
    """Write rate coefficient data to an ADF11-formatted file. Supports
    both metastable-unresolved and metastable-resolved output.

    Standard adf11 units:
        dens  in [cm-3]    (linear; converted to log10 internally)
        temp  in [eV]      (linear; converted to log10 internally)
        rates in [cm3 s-1] (linear; converted to log10 internally)

    Note: initial implementation generated with Claude (Anthropic),
    based on ADAS adf11 file format documentation.

    :param file_name: output file path
    :type str

    :param rates:
        UNRESOLVED MODE (metas=None):
            array of shape (nz, nt, nd), nz = izmax - izmin + 1,
            linear units.
        RESOLVED MODE (metas given):
            list of length nz. rates[iz] is an array of shape
            (n_child, n_parent, nt, nd), where n_child = metas[iz] (number of
            metastables in the child charge stage) and
            n_parent = metas[iz+1] (number of metastables in the parent
            charge stage), linear units.
    :type ndarray or list of ndarray

    :param nuclear_charge: nuclear charge Z0
    :type int

    :param izmin: minimum ion charge+1 (spectroscopic charge) in file
    :type int

    :param izmax: maximum ion charge in file
    :type int

    :param dens: density grid [cm-3], linear
    :type float arr

    :param temp: temperature grid [eV], linear
    :type float arr

    :param metas: if None, write metastable-unresolved format. Otherwise, a
                   list/array of length nz+1 giving the number of
                   metastable levels for charge states izmin..izmax+1.
    :type None or int arr

    :param rate_type: 3-letter ADAS data class, e.g. 'SCD'
    :type str

    :param reference: descriptive text (author/year/project) to be written in
                       file header line (max 22 characters).
    :type str

    :param date_str: date string for block headers; defaults to today
    :type str

    :param comment: dict of extra footer comment lines
    :type dict
    """

    nz = izmax - izmin + 1
    nd = len(dens)
    nt = len(temp)

    element = element_name(nuclear_charge)

    if rate_type not in ['SCD', 'ACD', 'QCD', 'XCD']:
        raise ValueError(f'Unsupported rate type {rate_type}')
    rate_short_names = {'SCD': 'IONIS', 'ACD': 'RECOM', 'QCD': '', 'XCD': ''}
    rate_short_name = rate_short_names[rate_type]

    if date_str is None:
        date_str = date.today().strftime('%d.%m.%y')

    log_dens = np.log10(dens)
    log_temp = np.log10(temp)

    resolved = metas is not None

    f = open(file_name, 'w+')

    # ---- header line: 5I5, 5X, '/', A20, '/', A24[, 'ADF11'] ----
    header_nums = ''.join('{:5d}'.format(v) for v in
                           (nuclear_charge, nd, nt, izmin, izmax))
    elem_field = '{:<20.20}'.format(element.upper())
    ref_field = '{:<22.22}'.format(reference)
    f.write(header_nums + '     /' + elem_field + '/' + ref_field + ' ADF11' + '\n')

    f.write('-' * 80 + '\n')

    if resolved:
        metas = list(metas)
        assert len(metas) == nz + 1, \
            "metas must have length nz+1 = {} (one metastable count " \
            "per charge state from izmin to izmax+1)".format(nz + 1)
        _write_int_block(f, metas)
        f.write('-' * 80 + '\n')

    # ---- density / temperature grids (shared, regardless of resolution) ----
    _write_float_block(f, log_dens)
    _write_float_block(f, log_temp)

    # ---- data blocks ----
    if not resolved:
        rates = np.asarray(rates)
        assert rates.shape == (nz, nt, nd), \
            "rates must have shape (izmax-izmin+1, len(temp), len(dens)), " \
            "got {}".format(rates.shape)
        log_rates = np.log10(np.clip(rates, a_min=1.e-74))

        for iz in range(nz):
            z1 = izmin + iz
            f.write(_block_header(rate_short_name, 1, 1, z1, date_str))
            for it in range(nt):
                _write_float_block(f, log_rates[iz, it, :])

    else:
        assert len(rates) == nz, \
            "in resolved mode, rates must be a list of length nz={}".format(nz)

        for iz in range(nz):
            z1 = izmin + iz
            n_child = metas[iz]
            n_parent = metas[iz + 1]
            block = np.asarray(rates[iz])
            assert block.shape == (n_child, n_parent, nt, nd), \
                "rates[{}] must have shape ({}, {}, {}, {}) based on " \
                "metas, got {}".format(iz, n_child, n_parent, nt, nd, block.shape)
            log_block = np.log10(np.clip(block, a_min=1.e-74))

            # Loop order: child (IGRD) outer, parent (IPRT) inner
            for igrd in range(1, n_child + 1):
                for iprt in range(1, n_parent + 1):
                    f.write(_block_header(rate_short_name, iprt, igrd, z1, date_str))
                    for it in range(nt):
                        _write_float_block(f, log_block[igrd - 1, iprt - 1, it, :])

    # ---- footer ----
    f.write('C' + '-' * 79 + '\n')
    f.write('C\n')
    f.write('C  {}: {}\n'.format(rate_type, element))
    f.write('C\n')
    f.write('C  INFORMATION\n')
    f.write('C  -----------\n')
    f.write('C\n')
    f.write('C   ***MADE WITH ColRadPy***\n')
    f.write('C  NUCLEAR CHARGE = {}\n'.format(nuclear_charge))
    f.write('C\n')

    if isinstance(comment, dict):
        for key in comment.keys():
            f.write('C ' + key + ' ' + str(comment[key]) + '\n')

    f.close()
