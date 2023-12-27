/*■■■■■■■■ 계약 데이터 산출 ■■■■■■■■■■■■■■■■■■■■■■*/
proc sql ;
 connect to saphana (&HanaCpc_Connect. );
 create table LIB.전체계약_202310 as
 select  *   from  connection to saphana (
           select a.ZA_FIN_YM as "마감년월"
     , a.za_cont_no as "계약번호"
     , a.ZA_CONTR_CUST_ID as "계약자고객ID"
     , a.ZA_CONTR_RRN_ENCR AS "계약자주민등록번호암호화"

     , a.ZA_CONT_YMD as "계약일자"
      , a.ZA_PRCD AS "상품코드"
      , a.ZA_PRDT_NM  AS "상품명"    

     , &product
     , &product2

              from   "_SYS_BIC"."LM.PM.M/ZCVPMM600"  as a    /* 마감계약 :PMM_계약스냅샷*/ 
    left join "_SYS_BIC"."LM.BP.B/ZCVMDB086" as b   /*MD-고객_기본*/
      on a.ZA_CONTR_CUST_ID=b.ZA_BP_NO

             where   a.ZA_FIN_YM = '202310'
                and    a.ZA_G_CONT_SC_CD <> '21'  /* 단체계약제외: 21=단체기업주계약*/
    and   ( a.ZA_CONT_UKEP_YN='X' or  A.ZA_CONT_STAT_CD  in  ('01') )
                and    a.ZA_FIN_SC_CD ='1'               /*마감기준_기본조건*/     
       and    b."BU_GROUP"='0101'  /*MD-고객_기본 : 그룹화 =0101 개인BP*/
 );
disconnect from saphana;
quit;




/*■■■■■■■■ 모델링 데이터 구성 ■■■■■■■■■■■■■■■■■■■■■■*/

%let 마감년월 = '202305' ;
%let i_마감년월 = %substr(&마감년월,2,6);
data _NULL_; call symput("AFT_6M",PUT(inTNX("MONTH",inPUT("&i_마감년월.",YYMMN6.),6),YYMMN6.)); RUN;
%PUT &AFT_6M.;

PROC SQL;
CREATE TABLE BASE_POP AS
SELECT DISTINCT A.*
FROM RCM.CUST_BASE_&i_마감년월 A
 LEFT JOIN (SELECT DISTINCT 계약자고객ID, 계약자주민등록번호암호화 FROM  LIB.전체계약_202310 WHERE SUBSTR(계약일자,1,6) > &마감년월 and SUBSTR(계약일자,1,6) <= put(&AFT_6M, 6.)) B
  ON A.계약자고객ID=B.계약자고객ID AND A.계약자주민등록번호암호화=B.계약자주민등록번호암호화
 WHERE B.계약자고객ID IS NOT NULL
;QUIT;
DATA BASE_POP ;
SET BASE_POP;
ID = _N_;
RUN;
proc sql; select count(계약자고객ID), count(distinct 계약자고객ID) from BASE_POP;quit;

/*■ 1. 고객 데이터 */
DATA LIB.CUST_BASE_&i_마감년월;
SET BASE_POP(DROP=계약자고객ID 계약자주민등록번호암호화);
RUN;

/*■ 맵핑 테이블 */
DATA LIB.MAPPING_&i_마감년월; 
SET  BASE_POP(KEEP=계약자고객ID 계약자주민등록번호암호화 ID);
RUN;

/*■ 2. 계약 데이터 */
PROC SQL;
CREATE TABLE LIB.PROD_BASE_&i_마감년월 AS
 SELECT BASE_POP.ID, A.상품코드, A.상품분류, A.상품중분류2, A.판매상품코드,  A.계약일자
 FROM  BASE_POP
  LEFT JOIN (SELECT * FROM LIB.전체계약_202310 WHERE SUBSTR(계약일자,1,6) <= put(&AFT_6M, 6.)) A
    ON BASE_POP.계약자고객ID=A.계약자고객ID AND BASE_POP.계약자주민등록번호암호화=A.계약자주민등록번호암호화
;QUIT;


  
/*■■■■■■■■ 적용 데이터 구성 ■■■■■■■■■■■■■■■■■■■■■■*/

%let 마감년월 = '202308' ;
%let i_마감년월 = %substr(&마감년월,2,6);
data _NULL_; call symput("AFT_6M",PUT(inTNX("MONTH",inPUT("&i_마감년월.",YYMMN6.),6),YYMMN6.)); RUN;
%PUT &AFT_6M.;

proc sql; select count(계약자고객ID), count(distinct 계약자고객ID) from RCM.CUST_BASE_&i_마감년월;quit;

DATA CUST_DATA;
SET RCM.CUST_BASE_&i_마감년월(KEEP = 계약자고객ID 계약자주민등록번호암호화
마감년월
계약자성별 외국인여부 BP상태코드 컨설턴트여부 임직원여부 관심고객여부 VIP등급 우량직종여부 직업대분류 직업군_관계사공통기준 투자성향 업종1 업종2
계약자연령 추정소득 최근계약경과월 F00003 F00004 F00005 F00006 F00007 F00008 F00009 F00010 F00011 F00012
);
ID = _N_;
RUN;

proc sql; select count(계약자고객ID), count(distinct 계약자고객ID) from CUST_DATA;quit;
/*■ 1. 고객 데이터 */
DATA LIB.CUST_APPLIED_&i_마감년월;
SET CUST_DATA(DROP=계약자고객ID 계약자주민등록번호암호화);
RUN;

/*■ 맵핑 테이블 */
DATA LIB.MAPPING_APPLIED_&i_마감년월; 
SET  CUST_DATA(KEEP=계약자고객ID 계약자주민등록번호암호화 ID);
RUN;

/*■ 2. 계약 데이터 */
PROC SQL;
CREATE TABLE PROD AS
 SELECT CUST_DATA.ID, A.상품코드, A.상품분류, A.상품중분류2, A.판매상품코드,  A.계약일자
 FROM  CUST_DATA
  LEFT JOIN (SELECT * FROM LIB.전체계약_202310 WHERE SUBSTR(계약일자,1,6) <= put(&AFT_6M, 6.)) A
    ON CUST_DATA.계약자고객ID=A.계약자고객ID AND CUST_DATA.계약자주민등록번호암호화=A.계약자주민등록번호암호화
 WHERE A.상품코드 IS NOT NULL
ORDER BY CUST_DATA.ID, A.계약일자 DESC
;QUIT;

DATA LIB.PROD_APPLIED_&i_마감년월;
SET PROD;
BY ID;
RETAIN ORDER_COLUMN;
IF FIRST.ID THEN ORDER_COLUMN=1;
ELSE ORDER_COLUMN + 1;
IF ORDER_COLUMN<=5;
RUN;











/*■■■■■■■■ OLE MODEL DATA ■■■■■■■■■■■■■■■■■■■■■■*/
PROC SQL;
CREATE TABLE LIB.OLD_202302 AS
 SELECT A.ID, B.추천상품 AS OLD
 FROM LIB.MAPPING_202302 A
  LEFT JOIN RCM._2302 B ON A.계약자고객ID=B.계약자고객ID
 WHERE B.계약자고객ID IS NOT NULL
;QUIT;
proc sql; select count(ID), count(distinct ID) from LIB.OLD_202302;quit;

PROC SQL;
CREATE TABLE LIB.OLD_202305 AS
 SELECT A.ID, B.추천상품 AS OLD
 FROM LIB.MAPPING_202305 A
  LEFT JOIN RCM._2305 B ON A.계약자고객ID=B.계약자고객ID
 WHERE B.계약자고객ID IS NOT NULL
;QUIT;
proc sql; select count(ID), count(distinct ID) from LIB.OLD_202305;quit;

PROC SQL;
CREATE TABLE LIB.OLD_202308 AS
 SELECT A.ID, B.추천상품 AS OLD
 FROM LIB.MAPPING_202308 A
  LEFT JOIN RCM._2308 B ON A.계약자고객ID=B.계약자고객ID
 WHERE B.계약자고객ID IS NOT NULL
;QUIT;
proc sql; select count(ID), count(distinct ID) from OLD_202308;quit;