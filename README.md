```ps1
ast-grep new
ast-grep scan
```

from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
from phopymnehelper.MNE_helpers import MNEHelpers


from phoofflineeeganalysis.analysis.historical_data import HistoricalData
from phopymnehelper.historical_data import HistoricalData


```ps1

ast-grep --lang python "PhoPyMNEHelper\src\phopymnehelper" --pattern 'from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers'

ast-grep --lang python "PhoPyMNEHelper\src\phopymnehelper" --pattern 'from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers' --rewrite 'from phopymnehelper.MNE_helpers import MNEHelpers' -U






ast-grep --lang python "PhoPyMNEHelper\src\phopymnehelper" --pattern 'from phoofflineeeganalysis.helpers.indexing_helpers' --rewrite 'from phopymnehelper.helpers.indexing_helpers' -U





ast-grep --lang python "PhoPyMNEHelper\src\phopymnehelper" --pattern 'from phoofflineeeganalysis.analysis.historical_data' --rewrite 'from phopymnehelper.historical_data' -U



```