import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/ranking_model.dart';
import '../services/api_service.dart';

final apiServiceProvider = Provider<ApiService>((ref) {
  return ApiService();
});

final rankingProvider =
    FutureProvider<List<RankingModel>>((ref) async {
  final api = ref.read(apiServiceProvider);
  return api.getRanking();
});