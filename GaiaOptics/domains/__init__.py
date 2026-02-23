from gaiaoptics.domains.microgrid.mission import build_problem_from_config as build_microgrid
from gaiaoptics.domains.data_center.mission import build_problem_from_config as build_data_center

DOMAIN_BUILDERS = {
    "microgrid": build_microgrid,
    "data_center": build_data_center,
}