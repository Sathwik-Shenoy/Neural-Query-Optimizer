SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id WHERE t1.value > 95.0 AND t1.category = 'A';
SELECT t1.id, t2.id FROM t1 JOIN t2 ON t1.join_id = t2.join_id JOIN t3 ON t1.join_id = t3.join_id WHERE t1.value > 110.0;
SELECT * FROM t2 JOIN t3 ON t2.join_id = t3.join_id WHERE t2.category = 'B';
