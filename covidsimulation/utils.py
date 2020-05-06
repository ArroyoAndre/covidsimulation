import sys
from datetime import date

if sys.version_info >= (3, 7):
    def get_date_from_isoformat(isoformat_date: str) -> date:
        return date.fromisoformat(isoformat_date)
else:
    import dateutil.parser


    def get_date_from_isoformat(isoformat_date: str) -> date:
        return dateutil.parser.isoparse(isoformat_date).date()
