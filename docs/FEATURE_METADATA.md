# BRFSS 2024 predictor coding check

Source: CDC codebook archive codebook24_llcp-v2-508.zip, member USCODE24_LLCP_082125.HTML, downloaded 2026-10-19. Hashes and dataset counts: DATA_PROVENANCE.json. All listed variable entries were inspected. This supersedes the earlier audit's outstanding codebook check for these 15 predictors only; it is not a review of every BRFSS variable.

| Variable | Meaning | Valid modeled values and treatment |
|---|---|---|
| CVDSTRK3 | Ever told had stroke | Target 1→1, 2→0; exclude 7, 9, blank. No incident-event or time-to-event label. |
| SEXVAR | Respondent sex | Categories 1, 2. Legacy defensive handling of 7/9 remains. |
| _AGEG5YR | Five-year age groups | Categories 1–13, including 80+; 14 missing. |
| GENHLTH | General health rating | Categories 1–5; 7/9 missing. Potential stroke-consequence correlate. |
| PHYSHLTH | Physically unhealthy days in past month | 1–30; 88→0; 77/99 missing. Potential stroke-consequence correlate. |
| MENTHLTH | Mentally unhealthy days in past month | 1–30; 88→0; 77/99 missing. |
| EXERANY2 | Leisure physical activity in past month | Categories 1/2; 7/9 missing. |
| _BMI5 | Calculated BMI, two implied decimals | Divide by 100. Blank missing. Legacy 0/9999 defensive missing handling retained; the BMI entry lists numeric range through 9999 whereas BMI-category notes treat 9999 as missing. This source inconsistency warrants sensitivity checking if actual 9999 values occur. |
| _SMOKER3 | Smoking status | Categories 1–4; 9 missing. |
| DIABETE4 | Diabetes history | Categories 1–4 retained separately, including pregnancy-only and prediabetes responses; 7/9 missing. |
| CVDINFR4 | Heart attack history | Categories 1/2; 7/9 missing. |
| CVDCRHD4 | Coronary disease/angina history | Categories 1/2; 7/9 missing. |
| CHCKDNY2 | Kidney disease history | Categories 1/2; 7/9 missing. |
| HAVARTH4 | Arthritis history | Categories 1/2; 7/9 missing. |
| EDUCA | Education | Categories 1–6; 9 missing. |
| INCOME3 | Income band | Categories 1–11 (7 and 9 are VALID income levels); only 77/99 missing. |

Categorical predictors use one-hot encoding. Numeric predictors use train-fitted median imputation and standardization; numeric missing indicators are included. Design fields _LLCPWT, _STSTR, _PSU, _STATE are retained separately and never predictors. Dropped target labels are not imputed.

Scope: these are survey reports, not imaging, laboratory tests, or real-time sensor measurements. The proposed acquisition engine must use realistic questionnaire-item acquisition terminology unless another dataset is added.
