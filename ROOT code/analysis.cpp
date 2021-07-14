struct pv {
  double pv;
  double pv03;
};

struct spe {
  double res;
  double peak;
  double peak_high;
  double peak_low;
};

spe find_spe_res_and_peak(TF1* f1, TF1* f2);
pv find_pv(TH1D* hist, spe peaks);

void analysis(string file, Double_t wind_st, Double_t wind_en,
		Int_t bins = 150, Double_t fit_low1 = -200,
		Double_t fit_high1 = 200, Double_t fit_low2 = 500,
              Double_t fit_high2 = 900, Double_t view_low = -2000,
              Double_t view_high = 3000, Bool_t log = true) {

	TFile* f = new TFile(file.c_str());
	TTree* t = (TTree*)f->Get("waveformTree");

	Double_t baseline, polarity, integral, timestamp;
	vector<UShort_t>* waveform = 0;
	t->SetBranchAddress("waveform", &waveform);
	t->SetBranchAddress("baseline", &baseline);
	t->SetBranchAddress("polarity", &polarity);
  t->SetBranchAddress("timestamp", &timestamp);

	t->GetEntry(0);
	Int_t tot_num = t->GetEntries();
	Double_t up_time = waveform->size() * 4.0 * tot_num; // convert to ns
	up_time /= 1e9; // convert to sec

  if (log) {
    auto c = new TCanvas("c", "c", 600, 400);
    c->SetLogy();
  }

	// histogram title
	string ext = ".root";
	file.resize(file.size() - ext.size());
  stringstream ss;
  ss << "Run " << file << " int. window " << wind_st << "-" << wind_en
    << ";Integrated ADC;Counts";
	TH1D* hist = new TH1D("hist", ss.str().c_str(), bins, view_low, view_high);

	for (Int_t i = 0; i < tot_num; i++) {
		integral = 0;
		t->GetEntry(i);
		for (Int_t j = wind_st; j < wind_en + 1; j++) {
			integral += (waveform->at(j) - baseline) *
				polarity;
      if (j % 1000 == 0) {
        cout << timestamp << " ";
      }
		}
		hist->Fill(integral);
	}

	// fit histogram with 2 gaussians
	TF1* fit_func1 = new TF1("fit1", "gaus", fit_low1, fit_high1);
	TF1* fit_func2 = new TF1("fit2", "gaus", fit_low2, fit_high2);
	hist->Fit(fit_func1, "RQ");
	hist->Fit(fit_func2, "RQ+");

	hist->Draw();

	spe photo_elec = find_spe_res_and_peak(fit_func1, fit_func2);
	pv peak_valley = find_pv(hist, photo_elec);

  ss.str("");
  ss << "P/V:\t\t" << peak_valley.pv << '\n'
    << "P/V (0.3SPE):\t" << peak_valley.pv03 << '\n'
    << "Resolution:\t" << photo_elec.res << '\n'
    << "SPE peak:\t" << photo_elec.peak;

  TPaveText* res = new TPaveText(0.5, 0.5, 0.75, 0.9, "NDC");
  ss.str("");
  ss << "P/V: " << peak_valley.pv;
  res->AddText(ss.str().c_str());
  ss.str("");
  ss << "Resolution: " << photo_elec.res;
  res->AddText(ss.str().c_str());
  ss.str("");
  ss << "SPE peak: " << photo_elec.peak;
  res->AddText(ss.str().c_str());
  res->Draw();

	// set axis back to normal
	hist->GetXaxis()->SetRangeUser(view_low, view_high);
}

pv find_pv(TH1D* hist, spe peaks) {
  pv to_return;
	hist->GetXaxis()->SetRangeUser(peaks.peak_low, peaks.peak_high);
  Double_t min = hist->GetMinimum();
  hist->GetXaxis()->SetRange(hist->GetMinimumBin(), 5000);
  Double_t max = hist->GetMaximum();
	Double_t pv = max / min;
	cout << "P/V:\t\t" << pv << endl;

	// PV for 0.3 SPE, from Hamamatsu spec sheet
	// taking 0.3 SPE to be 0.3 * peak SPE
	Double_t min_03 = hist->GetBinContent(hist->GetXaxis()->FindBin(max * 0.3));
	cout << "P/V (0.3SPE):\t" << max / min_03 << endl;
  to_return.pv = pv;
  to_return.pv03 = max / min_03;

  return to_return;
}

spe find_spe_res_and_peak(TF1* f1, TF1* f2) {
  spe to_return;
	Double_t mean1 = f1->GetParameter(1);
	Double_t mean2 = f2->GetParameter(1);
	Double_t var2 = f2->GetParameter(2);
	Double_t res = var2 / (mean2 - mean1);
	cout << "Resolution:\t" << res << endl;
	cout << "SPE peak:\t" << (mean2 - mean1) << endl;
  to_return.res = res;
  to_return.peak = (mean2 - mean1);
  to_return.peak_low = mean1;
  to_return.peak_high = mean2;

  return to_return;
}
