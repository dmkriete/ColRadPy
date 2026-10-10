# index 0 is a placeholder (None) so that ELEMENTS[z] matches atomic number z directly
ELEMENTS = [
    None,
    ('H', 'Hydrogen'), ('He', 'Helium'), ('Li', 'Lithium'), ('Be', 'Beryllium'),
    ('B', 'Boron'), ('C', 'Carbon'), ('N', 'Nitrogen'), ('O', 'Oxygen'),
    ('F', 'Fluorine'), ('Ne', 'Neon'), ('Na', 'Sodium'), ('Mg', 'Magnesium'),
    ('Al', 'Aluminium'), ('Si', 'Silicon'), ('P', 'Phosphorus'), ('S', 'Sulfur'),
    ('Cl', 'Chlorine'), ('Ar', 'Argon'), ('K', 'Potassium'), ('Ca', 'Calcium'),
    ('Sc', 'Scandium'), ('Ti', 'Titanium'), ('V', 'Vanadium'), ('Cr', 'Chromium'),
    ('Mn', 'Manganese'), ('Fe', 'Iron'), ('Co', 'Cobalt'), ('Ni', 'Nickel'),
    ('Cu', 'Copper'), ('Zn', 'Zinc'), ('Ga', 'Gallium'), ('Ge', 'Germanium'),
    ('As', 'Arsenic'), ('Se', 'Selenium'), ('Br', 'Bromine'), ('Kr', 'Krypton'),
    ('Rb', 'Rubidium'), ('Sr', 'Strontium'), ('Y', 'Yttrium'), ('Zr', 'Zirconium'),
    ('Nb', 'Niobium'), ('Mo', 'Molybdenum'), ('Tc', 'Technetium'), ('Ru', 'Ruthenium'),
    ('Rh', 'Rhodium'), ('Pd', 'Palladium'), ('Ag', 'Silver'), ('Cd', 'Cadmium'),
    ('In', 'Indium'), ('Sn', 'Tin'), ('Sb', 'Antimony'), ('Te', 'Tellurium'),
    ('I', 'Iodine'), ('Xe', 'Xenon'), ('Cs', 'Caesium'), ('Ba', 'Barium'),
    ('La', 'Lanthanum'), ('Ce', 'Cerium'), ('Pr', 'Praseodymium'), ('Nd', 'Neodymium'),
    ('Pm', 'Promethium'), ('Sm', 'Samarium'), ('Eu', 'Europium'), ('Gd', 'Gadolinium'),
    ('Tb', 'Terbium'), ('Dy', 'Dysprosium'), ('Ho', 'Holmium'), ('Er', 'Erbium'),
    ('Tm', 'Thulium'), ('Yb', 'Ytterbium'), ('Lu', 'Lutetium'), ('Hf', 'Hafnium'),
    ('Ta', 'Tantalum'), ('W', 'Tungsten'), ('Re', 'Rhenium'), ('Os', 'Osmium'),
    ('Ir', 'Iridium'), ('Pt', 'Platinum'), ('Au', 'Gold'), ('Hg', 'Mercury'),
    ('Tl', 'Thallium'), ('Pb', 'Lead'), ('Bi', 'Bismuth'), ('Po', 'Polonium'),
    ('At', 'Astatine'), ('Rn', 'Radon'), ('Fr', 'Francium'), ('Ra', 'Radium'),
    ('Ac', 'Actinium'), ('Th', 'Thorium'), ('Pa', 'Protactinium'), ('U', 'Uranium'),
    ('Np', 'Neptunium'), ('Pu', 'Plutonium'), ('Am', 'Americium'), ('Cm', 'Curium'),
    ('Bk', 'Berkelium'), ('Cf', 'Californium'), ('Es', 'Einsteinium'), ('Fm', 'Fermium'),
    ('Md', 'Mendelevium'), ('No', 'Nobelium'), ('Lr', 'Lawrencium'), ('Rf', 'Rutherfordium'),
    ('Db', 'Dubnium'), ('Sg', 'Seaborgium'), ('Bh', 'Bohrium'), ('Hs', 'Hassium'),
    ('Mt', 'Meitnerium'), ('Ds', 'Darmstadtium'), ('Rg', 'Roentgenium'), ('Cn', 'Copernicium'),
    ('Nh', 'Nihonium'), ('Fl', 'Flerovium'), ('Mc', 'Moscovium'), ('Lv', 'Livermorium'),
    ('Ts', 'Tennessine'), ('Og', 'Oganesson'),
]


def element_symbol(z):
    return ELEMENTS[z][0]


def element_name(z):
    return ELEMENTS[z][1]


def rate_interp_parse(rate_interp_str):
    """parse the possible interperolation rates


    :param rate_interp_str: The initial populations of levels at t=0
    :type rate_interp_str: str


    :returns: bool: log rate :interp_kind str
    """
    
    if(rate_interp_str[0:3] =='log'):
        log_rate = True
    else:
        log_rate = False

    interp_kind = 'slinear'
    if('linear' in rate_interp_str):
        interp_kind = 'linear'
    if('nearest' in rate_interp_str):
        interp_kind = 'nearest'
    if('zero' in rate_interp_str):
        interp_kind = 'zero'
    if('slinear' in rate_interp_str):
        interp_kind = 'slinear'
    if('quadratic' in rate_interp_str):
        interp_kind = 'quadratic'
    if('cubic' in rate_interp_str):
        interp_kind = 'cubic'

    return log_rate, interp_kind
