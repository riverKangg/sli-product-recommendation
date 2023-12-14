/*■■■■■■■■ 계약 데이터 산출 ■■■■■■■■■■■■■■■■■■■■■■*/
proc sql ;
 connect to saphana (&HanaCpc_Connect. );
 create table LIB.계약HIST as
 select  *   from  connection to saphana (
           select  DISTINCT
      SUBSTR(a.ZA_CONT_YMD,1,6) AS "계약일자"
     , a.ZA_CONTR_CUST_ID as "계약자고객ID"
        , &product
     , &product2
              from   "_SYS_BIC"."LM.PM.M/ZCVPMM600"  as a           /* 마감계약 :PMM_계약스냅샷*/
    left join "_SYS_BIC"."LM.BP.B/ZCVMDB086" as b
      on a.ZA_CONTR_CUST_ID=b.ZA_BP_NO /*MD-고객_기본*/
              /* 기본속성 조건*/
             where  a.ZA_CONT_YMD >= 20220901
                and    a.ZA_G_CONT_SC_CD <> '21'  /* 단체계약제외: 21=단체기업주계약*/
    and   ( a.ZA_CONT_UKEP_YN='X' or  A.ZA_CONT_STAT_CD  in  ('01') )
                and    a.ZA_FIN_SC_CD ='1'               /*마감기준_기본조건*/
       and    b."BU_GROUP"='0101'  /*MD-고객_기본 : 그룹화 =0101 개인BP*/
 );
disconnect from saphana;
quit;

/*■■■■■■■■ 월별 계약 현황  - 전체 ■■■■■■■■■■■■■■■■■■■■■■■*/
PROC SQL;
CREATE TABLE 월별계약현황DATA AS
 SELECT 계약일자, 상품중분류2, COUNT( DISTINCT 계약자고객ID) AS 고객수
 FROM LIB.계약hist
 WHERE 상품분류 NOT IN ('80.미니보험')
 GROUP BY 1,2
 ORDER BY 2,1
; QUIT;
PROC TRANSPOSE DATA=월별계약현황DATA OUT=월별계약현황(DROP=_NAME_);
 BY  상품중분류2;
 ID 계약일자 ;
 VAR 고객수;
;RUN;
PROC SQL;
CREATE TABLE 월별계약현황_비중 AS
 SELECT  상품중분류2
  , ROUND('202209'N/SUM( '202209'N)*100 , .2) AS '202209'N
  , ROUND('202210'N/SUM( '202210'N)*100 , .2) AS '202210'N
  , ROUND('202211'N/SUM( '202211'N)*100 , .2) AS '202211'N
  , ROUND('202212'N/SUM( '202212'N)*100 , .2) AS '202212'N
  , ROUND('202301'N/SUM( '202301'N)*100 , .2) AS '202301'N
  , ROUND('202302'N/SUM( '202302'N)*100 , .2) AS '202302'N
  , ROUND('202303'N/SUM( '202303'N)*100 , .2) AS '202303'N
  , ROUND('202304'N/SUM( '202304'N)*100 , .2) AS '202304'N
  , ROUND('202305'N/SUM( '202305'N)*100 , .2) AS '202305'N
  , ROUND('202306'N/SUM( '202306'N)*100 , .2) AS '202306'N
  , ROUND('202307'N/SUM( '202307'N)*100 , .2) AS '202307'N
  , ROUND('202308'N/SUM( '202308'N)*100 , .2) AS '202308'N
  , ROUND('202309'N/SUM( '202309'N)*100 , .2) AS '202309'N
  , ROUND('202310'N/SUM( '202310'N)*100 , .2) AS '202310'N
  , ROUND('202311'N/SUM( '202311'N)*100 , .2) AS '202311'N
 FROM 월별계약현황
 ORDER BY '202305'N DESC
; QUIT;
PROC DELETE DATA=월별계약현황DATA 월별계약현황; QUIT;

/*■■■■■■■■ 월별 계약 현황  - 대상자 ■■■■■■■■■■■■■■■■■■■■■*/
PROC SQL;
CREATE TABLE 월별계약현황DATA_대상자 AS
 SELECT 계약일자, 상품중분류2, COUNT( DISTINCT A.계약자고객ID) AS 고객수
 FROM LIB.계약hist A
  LEFT JOIN RCM.CUST_BASE_202305 B ON A.계약자고객ID=B.계약자고객ID
 WHERE 상품분류 NOT IN ('80.미니보험') AND B.계약자고객ID IS NOT NULL
 GROUP BY 1,2
 ORDER BY 2,1
; QUIT;
PROC TRANSPOSE DATA=월별계약현황DATA_대상자 OUT=월별계약현황_대상자(DROP=_NAME_);
 BY  상품중분류2;
 ID 계약일자 ;
 VAR 고객수;
;RUN;
PROC SQL;
CREATE TABLE 월별계약현황_대상자_비중 AS
 SELECT 상품중분류2
  , ROUND('202209'N/SUM( '202209'N)*100 , .2) AS '202209'N
  , ROUND('202210'N/SUM( '202210'N)*100 , .2) AS '202210'N
  , ROUND('202211'N/SUM( '202211'N)*100 , .2) AS '202211'N
  , ROUND('202212'N/SUM( '202212'N)*100 , .2) AS '202212'N
  , ROUND('202301'N/SUM( '202301'N)*100 , .2) AS '202301'N
  , ROUND('202302'N/SUM( '202302'N)*100 , .2) AS '202302'N
  , ROUND('202303'N/SUM( '202303'N)*100 , .2) AS '202303'N
  , ROUND('202304'N/SUM( '202304'N)*100 , .2) AS '202304'N
  , ROUND('202305'N/SUM( '202305'N)*100 , .2) AS '202305'N
  , ROUND('202306'N/SUM( '202306'N)*100 , .2) AS '202306'N
  , ROUND('202307'N/SUM( '202307'N)*100 , .2) AS '202307'N
  , ROUND('202308'N/SUM( '202308'N)*100 , .2) AS '202308'N
  , ROUND('202309'N/SUM( '202309'N)*100 , .2) AS '202309'N
  , ROUND('202310'N/SUM( '202310'N)*100 , .2) AS '202310'N
  , ROUND('202311'N/SUM( '202311'N)*100 , .2) AS '202311'N
 FROM 월별계약현황_대상자
 ORDER BY '202305'N DESC
; QUIT;
PROC DELETE DATA=월별계약현황DATA_대상자 월별계약현황_대상자; QUIT;
/*■■■■■■■■ 최종 결과 확인 ■■■■■■■■■■■■■■■■■■■■■*/
PROC PRINT DATA=월별계약현황_비중; RUN;
PROC PRINT DATA=월별계약현황_대상자_비중; RUN;



/*■■■ [참고] 상품명 확인 ■■■■■■■■■■■■■■■■■■■■*/
proc sql ;
 connect to saphana (&HanaCpc_Connect. );
 create table TMP as
 select  *   from  connection to saphana (
           select  DISTINCT
      SUBSTR(a.ZA_CONT_YMD,1,6) AS "계약일자"
     , a.ZA_CONTR_CUST_ID as "계약자고객ID"
     , a."ZA_PRDT_NM"  AS "상품명"
        , &product
     , &product2
              from   "_SYS_BIC"."LM.PM.M/ZCVPMM600"  as a           /* 마감계약 :PMM_계약스냅샷*/
    left join "_SYS_BIC"."LM.BP.B/ZCVMDB086" as b
      on a.ZA_CONTR_CUST_ID=b.ZA_BP_NO /*MD-고객_기본*/
              /* 기본속성 조건*/
             where  SUBSTR(a.ZA_CONT_YMD,1,6) = '202311'
                and    a.ZA_G_CONT_SC_CD <> '21'  /* 단체계약제외: 21=단체기업주계약*/
    and   ( a.ZA_CONT_UKEP_YN='X' or  A.ZA_CONT_STAT_CD  in  ('01') )
                and    a.ZA_FIN_SC_CD ='1'               /*마감기준_기본조건*/
       and    b."BU_GROUP"='0101'  /*MD-고객_기본 : 그룹화 =0101 개인BP*/
/*■어린이*/
    and ( a.ZA_PRDT_LCLSF_CD = 'A' and a.ZA_PRDT_MCLSF_CD2 = 'AZ' and a.ZA_PRDT_MCLSF_CD = 'AD')
/*■건강*/
/*    AND ((a."ZA_PRDT_LCLSF_CD" = 'A') and (a."ZA_PRDT_MCLSF_CD2" = 'AZ') and (a."ZA_PRDT_MCLSF_CD" in ( 'AE' , 'AF' )) and a."ZA_PRDT_SCLSF_CD" <>'AFJ1')*/
/*    and a."ZA_SALE_PRCD"  not in ('P0554','P0275' , 'P0396' , 'P0523' , 'P0525','P0397' ,'P0564','P0566','P0587', 'P0639', 'P0640', 'P0641', 'P0643' , 'P0644' , 'P0645'  ) */
 );
disconnect from saphana;
quit;

<Mail 첨부 파일>