from gaiaoptics.domains.microgrid.mission import build_problem_from_config as build_microgrid
from gaiaoptics.domains.data_center.mission import build_problem_from_config as build_data_center
from .warehouse_fleet.mission import build_problem as build_warehouse_fleet_problem
from gaiaoptics.core.registry import register_domain

register_domain("warehouse_fleet", build_warehouse_fleet_problem)

DOMAIN_BUILDERS = {
    "microgrid": build_microgrid,
    "data_center": build_data_center,
    "warehouse_fleet": build_warehouse_fleet_problem,
}