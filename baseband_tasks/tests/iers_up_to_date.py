# Needed for astropy <= 3.2, which do not have up-to-date IERS_B
# and have the wrong URL for IERS_A.
from astropy.utils.data import clear_download_cache, is_url_in_cache
from astropy.utils import iers

iers.conf.auto_download = False


def get_iers_up_to_date(time):
    # Inspired by pint.erfautils.iers_b_up_to_date.
    iers_b = iers.IERS_B.open()
    if iers_b[-1]["MJD"].value < time.mjd:
        might_be_old = is_url_in_cache(iers.IERS_B_URL)
        iers_b = iers.IERS_B.open(iers.IERS_B_URL, cache=True)
        if might_be_old and iers_b[-1]["MJD"].value < time.mjd:
            # Try wiping the download and re-downloading
            clear_download_cache(iers.IERS_B_URL)
            iers_b = iers.IERS_B.open(iers.IERS_B_URL, cache=True)
    if iers_b[-1]["MJD"].value < time.mjd:
        raise iers.IERSRangeError("could not update IERS")

    iers.IERS.iers_table = iers.IERS_Auto.iers_table = iers_b
    return iers_b
