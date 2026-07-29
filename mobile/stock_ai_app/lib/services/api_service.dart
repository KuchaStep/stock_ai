import 'package:dio/dio.dart';
import '../models/ranking_model.dart';

class ApiService {
  final Dio _dio = Dio(
    BaseOptions(
      baseUrl: "http://127.0.0.1:8000",
    ),
  );

  Future<List<RankingModel>> getRanking() async {
    final response = await _dio.get("/ranking");

    return (response.data as List)
        .map((e) => RankingModel.fromJson(e))
        .toList();
  }
}