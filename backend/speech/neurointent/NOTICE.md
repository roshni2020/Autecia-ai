# Third-party notice

`pipeline.py` and `utils.py` in this folder are
copied (pipeline.py: one line changed to make the Whisper model/threads configurable, marked `EchoLoop:`) from **Autiz-NeuroIntent** by devalshah-04
(https://github.com/devalshah-04/Autiz-NeuroIntent), whose README declares the MIT License.
At the time of copying the repository contained no LICENSE file; the copy relies on that
declaration. Copyright remains with the original author.

EchoLoop uses the speech-processing parts (audio conversion, Whisper, openSMILE GeMAPS,
RoBERTa embedding, the content/prosody fusion layer and dataset preprocessing) through
`adapter.py`. The hiring-related intent classes, scores and interpretations are not used.
