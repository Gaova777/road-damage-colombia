from typing import Protocol

from road_damage_colombia.domain.damage import Detection


class Detector(Protocol):
    name: str

    def detect(self, image) -> list[Detection]: ...
