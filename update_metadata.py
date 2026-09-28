import os
import re

rohan_metadata_path = r"d:\5g_timegrad\thesis_rohan_v2\tex\metadata.tex"
rishi_metadata_path = r"d:\5g_timegrad\thesis_rishi_v2\tex\metadata.tex"

rohan_content = r"""\newcommand{\thesistitle}{Comparative Study of DDPM and TimeGrad Predictors for Proactive 5G Handover}
\newcommand{\studentname}{Rohan Sharma}
\newcommand{\rollnumber}{243010108}
\newcommand{\degree}{Master of Technology}
\newcommand{\department}{Department of Data Science and Artificial Intelligence}
\newcommand{\specialization}{Data Science and Artificial Intelligence}
\newcommand{\supervisorname}{Dr. Mallikharjuna Rao K.}
\newcommand{\supervisordesignation}{Assistant Professor}
\newif\ifhastwosupervisors
\hastwosupervisorsfalse
\newcommand{\secondsupervisorname}{}
\newcommand{\secondsupervisordesignation}{}
\newcommand{\institutename}{IIIT Naya Raipur}
\newcommand{\institutecity}{Naya Raipur, Chhattisgarh, India, 493661}
\newcommand{\submissionmonthyear}{June 2026}
\newcommand{\submissionyear}{2026}
\newcommand{\ackplace}{Naya Raipur}
\newcommand{\authorbiodegree}{B.Tech}
\newcommand{\authorbiouniversity}{IIIT Naya Raipur}
\newcommand{\authoremail}{rohan24300@iiitnr.edu.in}
"""

rishi_content = r"""\newcommand{\thesistitle}{High-Speed Generative Diffusion and DDIM Deterministic Sampling for Risk-Aware Handover in 5G Networks}
\newcommand{\studentname}{Rishi Thakur}
\newcommand{\rollnumber}{243010109}
\newcommand{\degree}{Master of Technology}
\newcommand{\department}{Department of Data Science and Artificial Intelligence}
\newcommand{\specialization}{Data Science and Artificial Intelligence}
\newcommand{\supervisorname}{Dr. Srinivasa KG}
\newcommand{\supervisordesignation}{Professor}
\newif\ifhastwosupervisors
\hastwosupervisorstrue
\newcommand{\secondsupervisorname}{Dr. Mallikharjuna Rao K.}
\newcommand{\secondsupervisordesignation}{Assistant Professor}
\newcommand{\institutename}{IIIT Naya Raipur}
\newcommand{\institutecity}{Naya Raipur, Chhattisgarh, India, 493661}
\newcommand{\submissionmonthyear}{June 2026}
\newcommand{\submissionyear}{2026}
\newcommand{\ackplace}{Naya Raipur}
\newcommand{\authorbiodegree}{B.Tech}
\newcommand{\authorbiouniversity}{IIIT Naya Raipur}
\newcommand{\authoremail}{rishi24300@iiitnr.edu.in}
"""

with open(rohan_metadata_path, 'w') as f:
    f.write(rohan_content)

with open(rishi_metadata_path, 'w') as f:
    f.write(rishi_content)

print("Metadata written successfully.")
