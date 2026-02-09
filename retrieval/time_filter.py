

from datetime import datetime
from typing import Dict, List, Union


class TimeFilter:
    def __init__(self, tolerance_days: int = 1):
        self.tolerance_days = tolerance_days

    def filter(self, triples: List[Dict], dates: List[Union[Dict, str]]) -> List[Dict]:
        # If there is no temporal constraint, do not filter.
        if not dates:
            return triples

        normalized = self._normalize_dates(dates)
        if not normalized:
            return triples

        filtered: List[Dict] = []
        for triple in triples:
            triple_date = triple.get("date", "")
            if not triple_date:
                continue

            for date_info in normalized:
                if self._matches(triple_date, date_info, tolerance_days=self.tolerance_days):
                    filtered.append(triple)
                    break

        return filtered

    def _normalize_dates(self, dates: List[Union[Dict, str]]) -> List[Dict]:
        result: List[Dict] = []
        for d in dates:
            if isinstance(d, dict):
                dd = dict(d)
                # If format missing, detect it from date string
                if "format" not in dd and "date" in dd and isinstance(dd["date"], str):
                    dd["format"] = self._detect_format(dd["date"])
                result.append(dd)
            elif isinstance(d, str):
                fmt = self._detect_format(d)
                result.append({"date": d, "format": fmt})
        return result

    def _detect_format(self, date_str: str) -> str:
        if len(date_str) == 10 and date_str.count("-") == 2:
            return "iso"
        if len(date_str) == 7 and date_str.count("-") == 1:
            return "month_year"
        if len(date_str) == 4 and date_str.isdigit():
            return "year"
        return "unknown"

    def _matches(self, triple_date: str, date_info: Dict, tolerance_days: int) -> bool:
        query_date = date_info.get("date")
        date_format = date_info.get("format")

        if not triple_date or not query_date:
            return False

        if date_format == "iso":
            return self._within_tolerance(triple_date, query_date, tolerance_days)

        if date_format in {"month_year", "year"}:
            # "2014-08-16".startswith("2014-08") or .startswith("2014")
            return triple_date.startswith(query_date)

        return False

    def _within_tolerance(self, d1: str, d2: str, tolerance_days: int) -> bool:
        try:
            date1 = datetime.strptime(d1, "%Y-%m-%d")
            date2 = datetime.strptime(d2, "%Y-%m-%d")
            return abs((date1 - date2).days) <= tolerance_days
        except ValueError:
            return False
