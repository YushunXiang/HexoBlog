---
title: 西北工业大学数据库系统实验——练习1
date: 2022-05-24 02:47:39
tags: 课程学习
categories: 数据库系统
---

==SQL Exercises Answer==

姓名：项裕顺

学号：2020302877

班级：14012003

## Create Tables

```
department(dNo,dName,officeRoom,homepage)
student(sNo,sName,sex,age,dNo)
course(cNo,cName,cPNo,credit,dNo)
sc(sNo,cNo,score,recordDate)
```



## Query Operations

### 一、单表

（1）查询所有年龄大于等于20岁的学生学号、姓名；

``` sql
SELECT sNo, sName
FROM student
WHERE age >= 20;
```



（2）查询所有姓钱的男生学号、姓名、出生年份；

``` sql
SELECT sNo, sName, EXTRACT(YEAR FROM CURRENT_DATE) - age AS sBirthday
FROM student
WHERE sName LIKE '钱%'
  AND sex = '男';
```



（3）查询所有学分大于3的课程名称；

``` sql
SELECT cName
FROM course
WHERE credit > 3;
```



（4）查询所有没有被分配到任何学院的学生姓名；

``` sql
SELECT sName
FROM student
WHERE dNo IS null;
```



（5）查询所有尚未设置主页的学院名称。

``` sql
SELECT dName
FROM department
WHERE homepage IS null;
```



### 二、聚集

（1）查询各个学院的平均年龄；

``` sql
SELECT department.dName AS 学院名称, ROUND(AVG(student.age), 2) AS 平均年龄
FROM student, department
GROUP by student.dNo, department.dNo
HAVING student.dNo = department.dNo;
```



（2）查询每个学生选修课程的平均分；

``` sql
SELECT student.sName AS 学生姓名, ROUND(AVG(sc.score), 2) AS 选修课程平均分
FROM student,
     sc
GROUP BY sc.sNo, student.sNo
HAVING sc.sNo = student.sNo;
```



（3）查询各课程的平均分；

``` sql
SELECT cName AS 课程名称, ROUND(AVG(score), 2) AS 课程平均分
FROM course,
     sc
GROUP BY course.cNo, sc.cNo
HAVING course.cNo = sc.cNo;
```



（4）查询各学院开设的课程门数；

> 注意开设课程数为0的学院也应该在查询结果中显示

``` sql
SELECT dName AS 学院名称, COUNT(cNO) AS 开课数
FROM department
         LEFT OUTER JOIN course
                         ON department.dNo = course.dNo
GROUP BY department.dNo;
```



（5）查询各门课程选修人数。

> 注意选修人数为0的课程也应该在查询结果中显示

``` sql
SELECT cName AS 课程名称, count(sNo) AS 选课人数
FROM sc
         RIGHT OUTER JOIN course
                          ON sc.cNo = course.cNo
GROUP BY course.cNo
```



### 三、多表
（1）查询“信息学院”所有学生学号与姓名；

``` sql
SELECT sNo AS 学号, sName AS 姓名
FROM student
WHERE student.dno = (
    SELECT dNo
    FROM department
    WHERE dName = '信息学院'
)
```



（2）查询“软件学院”开设的所有课程号与课程名称；

``` sql
SELECT cNo AS 课程号, cName AS 课程名称
FROM course
WHERE course.dNO = (
    SELECT dNo
    FROM department
    WHERE dName = '软件学院'
)
```



（3）查询与“陈丽”在同一个系的所有学生学号与姓名；

``` sql
SELECT sNo AS 学号, sName AS 姓名
FROM student st1
WHERE dNo IN (
    SELECT st2.dNO
    FROM student st2
    WHERE sName = '陈丽'
    LIMIT 1
)
  AND sName <> '陈丽';
```



（4）查询与“张三”同岁的所有学生学号与姓名；

``` sql
SELECT sNo AS 学号, sName AS 姓名
FROM student st1
WHERE age IN (
    SELECT st2.age
    FROM student st2
    WHERE st2.sName = '张三'
    LIMIT 1
)
  AND st1.sName <> '张三'
```



（5）查询与“张三”同岁且不与“张三”在同一个系的学生学号与姓名；

``` sql
SELECT sno AS 学号, sname AS 姓名
FROM student st1
WHERE age IN (
    SELECT st2.age
    FROM student st2
    WHERE st2.sName = '张三'
    LIMIT 1
)
  AND dno NOT IN (
    SELECT st2.dNo
    FROM student st2
    WHERE st2.sName = '张三'
    LIMIT 1
);
```



（6）查询学分大于“离散数学”的所有课程名称；

``` sql
SELECT cname AS 课程名称
FROM course c1
WHERE credit > (
    SELECT c2.credit
    FROM course c2
    WHERE c2.cname = '离散数学'
);

```



（7）查询选修了课程名为“组合数学”的学生人数；

``` sql
SELECT count(student.sno) AS 学生人数
FROM student
WHERE student.sno IN (
    SELECT sc.sno
    FROM sc
    WHERE sc.cno = (
        SELECT course.cno
        FROM course
        WHERE course.cname = '组合数学'
    )
);
```



（8）查询没有选修“离散数学”的学生姓名；

``` sql
SELECT student.sname AS 学生姓名
FROM student
WHERE student.sno NOT IN (
    SELECT sc.sno
    FROM sc
    WHERE sc.cno = (
        SELECT course.cno
        FROM course
        WHERE course.cname = '离散数学'
    )
);
```



（9）查询与“算法设计与分析”、“移动计算”学分不同的所有课程名称；

``` sql
SELECT cname AS 课程名称
FROM course
WHERE credit NOT IN (
    SELECT credit
    FROM course
    WHERE cname = '算法设计与分析'
       OR cname = '移动计算'
);
```



（10）查询平均分大于等于90分的所有课程名称；

``` sql
SELECT course.cname AS 课程名称
FROM course
WHERE course.cno IN (
    SELECT sc.cno
    FROM sc
    GROUP BY sc.cno
    HAVING AVG(sc.score) >= 80
);
```



（11）查询选修了“离散数学”课程的所有学生姓名与成绩；

``` sql
SELECT sName AS 姓名, score AS 离散数学成绩
FROM student
         RIGHT OUTER JOIN sc
                          ON student.sNo = sc.sNo
WHERE sc.cNo = (
    SELECT course.cNo
    FROM course
    WHERE cname = '离散数学'
)
```



（12）查询“王兵”所选修的所有课程名称及成绩；

``` sql
SELECT cName AS 课程名称, score AS 成绩
FROM course
         RIGHT OUTER JOIN sc
                          ON course.cNo = sc.cNo
WHERE sc.sNo IN (
    SELECT st.sNo
    FROM student st
    WHERE st.sName = '王兵'
    LIMIT 1
);
```



（13）查询所有具有不及格课程的学生姓名、课程名与成绩；

``` sql
SELECT sName AS 学生姓名, cName AS 课程名称, score AS 成绩
FROM student
         RIGHT OUTER JOIN sc
                          ON sc.sNo = student.sNo

         RIGHT OUTER JOIN course
                          ON course.cNo = sc.cNo

WHERE (
          score < 60
          )
```



（14）查询选修了“文学院”开设课程的所有学生姓名；

``` sql
SELECT sname
FROM student
WHERE sno IN (
    SELECT sc.sno
    FROM sc,
         course
    WHERE sc.cno = course.cno
      AND course.dno = (
        SELECT dno
        FROM department
        WHERE dname = '文学院'
    )
);
```



（15）查询“信息学院”所有学生姓名及其所选的“信息学院”开设的课程名称。

``` sql
SELECT student.sname, course.cname
FROM student,
     course,
     sc
WHERE student.dno = (
    SELECT dno
    FROM department
    WHERE dname = '信息学院'
)
  AND course.dno = (
    SELECT dno
    FROM department
    WHERE dname = '信息学院'
)
  AND sc.sno = student.sno
  AND sc.cno = course.cno;
```



### 四、综合
（1）查询所有学生及其选课信息（包括没有选课的学生）；

``` sql
SELECT *
FROM student
         LEFT OUTER JOIN sc
                         ON student.sNo = sc.sNo;
```



（2）查询“形式语言与自动机”先修课的课程名称；

``` sql
SELECT cname
FROM course
WHERE cno = (
    SELECT cpno
    FROM course
    WHERE cname = '形式语言与自动机'
);
```



（3）查询“形式语言与自动机”间接先修课课程名称；

``` sql
SELECT cname
FROM course
WHERE cno = (
    SELECT cpno
    FROM course
    WHERE cno = (
        SELECT cpno
        FROM course
        WHERE cname = '形式语言与自动机'
    )
);
```



（4）查询先修课为编译原理数学的课程名称

``` sql
SELECT second.cname
FROM course first,
     course second
WHERE first.cno = second.cpno
  AND first.cname = '编译原理数学'
```



（5）查询间接先修课为离散数学的课程名称；

```sql
SELECT third.cName
FROM course first,
	 course second,
     course third
WHERE first.cname = '离散数学'
  AND second.cno = third.cpno
  AND first.cno = second.cpno;
```



（6）查询所有没有先修课的课程名称；

``` sql
SELECT cname
FROM course
WHERE cpno IS NULL;
```



（7）查询所有没选修“形式语言与自动机”课程的学生姓名；

``` sql
SELECT sname
FROM student
WHERE sno NOT IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cno
        FROM course
        WHERE cname = '形式语言与自动机'
    )
);    
```



（8）查询仅仅选修了离散数学一门课程的学生姓名；

``` sql
SELECT sname
FROM student
WHERE student.sno IN (
    SELECT sno
    FROM sc
    GROUP BY sc.sno
    HAVING count(*) = 1
    )
  AND student.sno IN (
      SELECT sno
      FROM sc, course
      WHERE sc.cno = course.cno AND cname = '离散数学'
    )
```



（9）查询所有选修了“形式语言与自动机”但没选修其先修课的学生姓名；

``` sql
SELECT sname
FROM student
WHERE sno IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cno
        FROM course
        WHERE cname = '形式语言与自动机'
    )
)
  AND sno NOT IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cpno
        FROM course
        WHERE cname = '形式语言与自动机'
    )
);
```



（10）查询选修课程总学分大于等于28的学生姓名及其选修课程总学分；

``` sql
SELECT student.sname, count(course.credit) AS totalCredit
FROM student,
     course,
     sc
WHERE student.sno = sc.sno
  AND sc.cno = course.cno
GROUP BY student.sno
HAVING count(course.credit) >= 28;
```



（11）查询选修了3门以上课程且成绩都大于85分的学生学号与姓名；

``` sql
SELECT student.sno, student.sname
FROM student,
     sc
WHERE student.sno = sc.sno
GROUP BY student.sno
HAVING min(sc.score) > 85
   AND count(*) > 3;
```



（12）查询恰好选修了3门课并且都及格的学生姓名；

``` sql
SELECT student.sname
FROM student,
     sc
WHERE student.sno = sc.sno
GROUP BY student.sno
HAVING min(sc.score) >= 60
   AND count(*) = 3;
```



（13）查询人数多于6的学院名称及其学生人数；

``` sql
SELECT department.dname, count(*) AS studentNum
FROM department,
     student
WHERE department.dno = student.dno
GROUP BY department.dno
HAVING count(*) > 6;
```



（14）查询平均成绩高于王兵的学生姓名；

``` sql
SELECT student.sname
FROM student,
     sc
WHERE student.sno = sc.sno
GROUP BY student.sno
HAVING avg(sc.score) > (
    SELECT avg(sc.score)
    FROM student, sc
    WHERE student.sno = sc.sno
      AND student.sname = '王兵'
    GROUP BY student.sno
    LIMIT 1
);
```



（15）查询所有选修了离散数学并且选修了编译原理课程的学生姓名；

``` sql
SELECT sname
FROM student
WHERE sno IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cno
        FROM course
        WHERE cname = '离散数学'
    )
)
  AND sno IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cno
        FROM course
        WHERE cname = '编译原理'
    )
);
```



（16）查询软件学院离散数学课程平均分；

``` sql
SELECT avg(score) AS 平均分
FROM student,
     sc
WHERE student.sNo = sc.sNo
  AND sc.cNo IN (
    SELECT cNo
    FROM course
    WHERE cName = '离散数学'
)
  AND student.dNo IN (
    SELECT dNo
    FROM department
    WHERE dName = '软件学院'
);
```



（17）查询年龄与“软件学院”所有学生年龄都不相同学生姓名及其年龄和学院；

``` sql
SELECT student.sname, student.age, department.dname
FROM student
         LEFT OUTER JOIN department on student.dno = department.dno
WHERE student.age <> ALL (
    SELECT age
    FROM student
    WHERE dno = (
        SELECT dno
        FROM department
        WHERE dname = '软件学院'
    )
      AND age IS NOT NULL
);
```



（18）查询各学院选修同一门课人数大于4的学院、课程及选课人数；

``` sql
SELECT dp.dName, cs.cName, count(sc.sNo)
FROM department dp
         LEFT OUTER JOIN student st
                         ON st.dNo = dp.dNo
         LEFT OUTER JOIN sc
                         ON sc.sNo = st.sNo
         LEFT OUTER JOIN course cs
                         ON cs.cNo = sc.cNo

GROUP BY dp.dNo, cs.cNo
HAVING count(sc.sNo) > 4;
```



（19）查询仅仅选修了“高等数学”一门课程的学生姓名；（学号、姓名及所在学院名称）

``` sql
SELECT s.sno, s.sname, d.dname
FROM sc
         NATURAL JOIN student s
         LEFT JOIN department d ON s.dno = d.dno
WHERE s.sno IN (
    SELECT sno
    FROM sc
    GROUP BY sc.sno
    HAVING count(*) = 1
)
  AND s.sno IN (
    SELECT sno
    FROM sc,
         course
    WHERE sc.cno = course.cno
      AND cname = '高等数学'
);
```



（20）查询平均学分积小于70分的学生姓名。

``` sql
SELECT sname
FROM student
WHERE exists(
              SELECT student
              FROM sc,
                   course
              WHERE sc.cno = course.cno
                AND sc.sno = student.sno
                AND sc.score IS NOT NULL
              GROUP BY sc.sno
              HAVING sum(sc.score * course.credit) / sum(course.credit) < 70
          );
```



（21）查询选修了“信息学院”开设全部课程的学生姓名。

``` sql
SELECT sname
FROM student
WHERE NOT exists(
        SELECT *
        FROM course
        WHERE NOT exists(
                SELECT *
                FROM sc
                WHERE sno = student.sno
                  AND cno = course.cno
            )
          AND course.dno = (
            SELECT dno
            FROM department
            WHERE dname = '信息学院'
        )
    );
```



（21）查询选修了“杨佳伟”同学所选修的全部课程的学生姓名。

``` sql
SELECT sname
FROM student
WHERE NOT exists(
        SELECT *
        FROM course
        WHERE NOT exists(
                SELECT *
                FROM sc
                WHERE sno = student.sno
                  AND cno = course.cno
            )
          AND cno IN (
            SELECT sc.cno
            FROM sc
            WHERE sc.sno = (
                SELECT sno
                FROM student
                WHERE sname = '杨佳伟'
            )
        )
    )
  AND sname <> '杨佳伟';
```



## DDL

1、创建2张表，信息如下：
      图书（编号，书名，作者，ISBN，出版社编号，版本，出版日期）。主码为编号，ISBN唯一。出版社编号为外码，参照出版社编号。
      出版社（编号，名称，地址，电话）。主码为编号。

要求：

(1)创建表的同时创建约束；

``` sql
CREATE TABLE publisher
(
    publisher_no      VARCHAR(20)  NOT NULL UNIQUE,
    publisher_name    VARCHAR(200),
    publisher_address VARCHAR(200),
    publisher_tel     VARCHAR(20),
    PRIMARY KEY (publisher_no)
);

CREATE TABLE book
(
    book_no      VARCHAR(20)  NOT NULL UNIQUE,
    book_name    VARCHAR(200),
    book_author  VARCHAR(200),
    book_ISBN    VARCHAR(200) NOT NULL UNIQUE,
    publisher_no VARCHAR(20),
    book_version VARCHAR(10),
    book_date    DATE,
    PRIMARY KEY (book_no),
    FOREIGN KEY (publisher_no) REFERENCES publisher (publisher_no)
);
```



(2)删除所创建的表;

``` sql
DROP TABLE book CASCADE;
DROP TABLE publisher CASCADE;
```



(3)重新创建表，在表创建之后增加约束。

``` sql
CREATE TABLE publisher
(
    publisher_no      VARCHAR(20)  NOT NULL UNIQUE,
    publisher_name    VARCHAR(200),
    publisher_address VARCHAR(200),
    publisher_tel     VARCHAR(20)
);

CREATE TABLE book
(
    book_no      VARCHAR(20)  NOT NULL UNIQUE,
    book_name    VARCHAR(200),
    book_author  VARCHAR(200),
    book_ISBN    VARCHAR(200) NOT NULL UNIQUE,
    publisher_no VARCHAR(20),
    book_version VARCHAR(10),
    book_date    DATE
);

ALTER TABLE publisher
    ADD PRIMARY KEY (publisher_no);
ALTER TABLE book
    ADD PRIMARY KEY (book_no);
ALTER TABLE book
    ADD FOREIGN KEY (publisher_no) REFERENCES publisher (publisher_no);
```



2、

(1)分别向两张表中各插入2行数据。

``` sql
INSERT INTO publisher
VALUES ('001', '高等教育出版社', '北京市西城区德外大街4号', '82086060');

INSERT INTO publisher
VALUES ('002', '机械工业出版社', '北京市西城区百万庄大街22号', '88378991');

INSERT INTO book
VALUES ('001', '数据库系统概论', '王珊', '978-7-04-040664-1', '001', '5.0', to_date('2016-12-12', 'yyyy-mm-dd'));

INSERT INTO book
VALUES ('002', '计算机网络', 'James F. Kurose', '978-7-111-59971-5', '002', '7.0', to_date('2018-11-11', 'yyyy-mm-dd'));
```



(2)将其中一个出版社地址变更一下。

``` sql
UPDATE publisher
SET publisher_address = '上海浦东新区'
WHERE publisher_no = '001';
```



(3)删除所插入数据。

``` sql
DELETE FROM book;
DELETE FROM press;
```



3、
(1)创建一个软件学院所有选修了“离散数学”课程的学生视图，并通过视图插入一行数据。

``` sql
CREATE VIEW view_student
AS
SELECT sno, sname
FROM student
WHERE dno = (
    SELECT dno
    FROM department
    WHERE dname = '软件学院')
  AND sno IN (
    SELECT sno
    FROM sc
    WHERE cno = (
        SELECT cno
        FROM course
        WHERE cname = '离散数学'));
INSERT
INTO view_student
VALUES ('2020302877', '项裕顺');
```



(2)创建一个各门课程平均分视图。

``` sql
CREATE VIEW avg_score
AS
SELECT sc.cno, course.cname, avg(sc.score)
FROM sc
         LEFT OUTER JOIN course ON sc.cNo = course.cNo
GROUP BY sc.cno, course.cname;
```



4、创建一张学生平均成绩表s_score(sNo,sName,avgscore),并通过子查询插入所有学生数据。

``` sql
CREATE TABLE s_score
(
    sNo      CHAR(10),
    sName    VARCHAR(100),
    avgScore FLOAT
);

INSERT INTO s_score
SELECT s.sno, s.sname, avg(score)
FROM student s
         LEFT OUTER JOIN sc
                         ON s.sNo = sc.sNo
GROUP BY s.sno, s.sname;
```



## DCL
尝试将多条SQL语句组成一个事务执行，体验提交和回滚操作。

``` sql
BEGIN TRANSACTION;

DELETE FROM student WHERE sno = '2020302877';
DELETE FROM course WHERE course.cName= '数据库系统';
COMMIT;
ROLLBACK;
```