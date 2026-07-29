import 'package:dio/dio.dart';

class RankingService {
  final Dio _dio = Dio(
    BaseOptions(
      // Androidエミュレータなら 10.0.2.2
      // Webなら localhost
      baseUrl: "http://127.0.0.1:8000",
    ),
  );

  Future<List<dynamic>> getRanking() async {
    final response = await _dio.get("/ranking");
    return response.data;
  }
}