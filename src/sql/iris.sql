-- 1. 상품 전체
SELECT
 "ZA_PRCD" AS "상품코드"
 ,"ZA_PRDT_SC_PRMTR_VAL" AS "상품구분파라미터값"
 ,"ZA_PRDT_APLC_PRMTR_VAL" AS "상품적용파라미터값"
 ,"ZA_PRDT_APLC_PRMTR_NM" AS "상품적용파라미터명"
 ,"ZA_OLD_PRCD" AS "구상품코드"
 ,"ZA_PRDT_NM" AS "상품명"
 ,"ZA_PRDT_LCLSF_CD" AS "상품대분류코드"
 ,"ZA_PRDT_MCLSF_CD2" AS "상품중분류코드2"
 ,"ZA_PRDT_MCLSF_CD" AS "상품중분류코드"
 ,"ZA_PRDT_SCLSF_CD" AS "상품소분류코드"
 ,"ZA_SALE_BGN_YMD" AS "판매개시일자"
 ,"ZA_SALE_END_YMD" AS "판매종료일자"
FROM "ZADPAB114"



-- 2. 주보험 전체
SELECT
 "ZA_COLTR_CD" AS "담보코드"
 ,"ZA_COLTR_SC_PRMTR_VAL" AS "담보구분파라미터값"
 ,"ZA_COLTR_APLC_PRMTR_VAL" AS "담보적용파라미터값"
 ,"ZA_COLTR_APLC_PRMTR_NM" AS "담보적용파라미터명"
 ,"ZA_OLD_INSCD" AS "구보험코드"
 ,"ZA_EPNAM" AS "담보명"
 ,"ZA_MN_CONT_TRTY_SC_CD" AS "주계약특약구분코드"
 ,"ZA_PREFR_INSR_DTL_SC_CD" AS "우량체보험상세구분코드"
 ,"ZA_INSR_LCLSF_CD" AS "보험대분류코드"
 ,"ZA_INSR_MCLSF_CD" AS "보험중분류코드"
 ,"ZA_SALE_BGN_YMD" AS "판매개시일자"
FROM "ZADPAB208"
WHERE "ZA_MN_CONT_TRTY_SC_CD" = '1'



-- 3. 위험률 전체
SELECT DISTINCT
 "ZA_RISKR_CD" AS "위험율코드"
 ,"ZA_RISKR_NM" AS "위험율명"
 , "ZA_DTL_COLTR_CD" AS "상세담보코드"
 ,"ZA_MORTA_ACDT_TYP_CD" AS "사차사고유형코드"
 ,"ZA_VRT_RISKR_YN" AS "가상위험율여부"
 ,"ZA_RISKR_SC_CD" AS "위험율구분코드"
 ,"ZA_EXLTB_CD" AS "경험생명표코드"
 ,"ZA_DVD_ANUT_SC_CD" AS "배당연금구분코드"
 ,"ZA_MORTA_AGE_SC_CD" AS "사차연령구분코드"
 ,"ZA_APPR_YMD" AS "인가일자"
 ,"ZA_RISKR_BASE_CNTNT" AS "위험율근거내용"
 ,"ZA_MALE_WGT_SGN_CD" AS "남성가중치부호코드"
 ,"ZA_FEM_WGT_SGN_CD" AS "여성가중치부호코드"
 ,"ZA_MALE_WGT_VAL" AS "남성가중치값"
 ,"ZA_FEM_WGT_VAL" AS "여성가중치값"
 ,"ZA_EXMRP_DDS" AS "면책일수"
 ,"ZA_RISKR_OCCU_SC_CD" AS "위험율발생구분코드"
 ,"ZA_COVR_DIGN_CD" AS "보장진단코드"
 ,"ZA_COVR_CLAM_RSN_CD" AS "보장청구사유코드"
 ,"ZA_SURG_CLCD" AS "수술분류코드"
 ,"ZA_ANUL_COVR_NTS" AS "연간보장횟수"
 ,"ZA_INSR_ANUL_LMT_AMT" AS "연간한도금액(보험)"
 ,"ZA_ANUL_LMT_AMT" AS "연간한도금액(청구)"
 ,"ZA_ANUL_LMT_STND_EXCD_AMT" AS "연간한도기준초과금액"
 ,"ZA_ANUL_LMT_NTS" AS "연간한도횟수"
 ,"ZA_PRMNC_CURE_TOOTH_SC_CD" AS "영구치유치구분코드"
 ,"ZA_HSPZ_RST_SC_CD" AS "입원요양구분코드"
 ,"ZA_DSAB_1G_RAT" AS "장해1급율"
 ,"ZA_FCDSA_KND_NM" AS "장해1급종류명"
 ,"ZA_FCDSA_ICLU_YN" AS "장해1급포함여부"
 ,"ZA_DSAB_RAATE_SC_CD" AS "장해률구분코드"
 ,"ZA_RSTC_DDS" AS "제한일수"
 ,"ZA_JOB_JSRS_CD" AS "직업직종코드"
 ,"ZA_DISE_SC_CD" AS "질병구분코드"
 ,"ZA_REFR_RR_CD" AS "참조위험률코드"
 ,"ZA_DENT_RMDY_MTH_SC_CD" AS "치과치료방법구분코드"
 ,"ZA_MDHSP_NTS" AS "통원횟수"
 ,"ZA_RLDM_LMT_AMT" AS "실손한도금액"
 ,"ZA_LMT_RTO" AS "한도비율"
 ,"ZA_LMT_DDS" AS "한도일수"
 ,"ZA_HOLDY_SC_CD" AS "공휴일구분코드"
 ,"ZA_DLYAL_TTW_LMT_AMT" AS "일당만원한도금액"
 ,"ZA_RLDM_SBTR_AMT" AS "실손공제금액"
 ,"ZA_SBTR_DDS" AS "공제일수"
 ,"ZA_RISKR_ETC_COND_CD" AS "위험율기타조건코드"
 ,"ZA_COVR_OBJ_MDCR_RTO_SARY" AS "보장대상의료비율(급여)"
 ,"ZA_COVR_OBJ_MDCR_RTO_NSLRY" AS "보장대상의료비율(비급여)"
 ,"ZA_OBNGC_ICLU_SC_CD" AS "산부인과포함구분코드"
 ,"ZA_UPER_WARD_DFAMT_RTO" AS "상급병실차액비율"
 ,"ZA_NEW_HSPZ_STND_VAL" AS "신규입원기준값"
FROM "ZADPAA204"




-- 4. 상품+담보
SELECT DISTINCT
"ZA_PRCD" AS "상품코드"
,"ZA_COLTR_CD" AS "담보코드"
FROM "ZADPAA604"
WHERE "ZA_MN_CONT_TRTY_SC_CD"='1'
AND "ZA_PRCD"<>'' AND "ZA_COLTR_CD"<>''



-- 5. 담보+위험률

SELECT DISTINCT
"ZA_COLTR_CD" AS "담보코드"
-- ,"ZA_RISKR_ID" AS "위험율ID"
,"ZA_RISKR_CD1" AS "위험율코드"
FROM "ZADPAB391"