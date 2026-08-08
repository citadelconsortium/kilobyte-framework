import unittest
from pathlib import Path

from kilobyte.config import MODEL_SHA256, MODEL_URL


class InstallationTests(unittest.TestCase):
    def test_model_is_pinned_and_atomic_installer(self):
        if not MODEL_URL:
            self.skipTest("kilobyte-framework ships without a bundled brain")
        script = (Path(__file__).parents[1] / "scripts" / "install-model.sh").read_text()
        self.assertIn(MODEL_SHA256, script)
        self.assertIn(".part", script)
        self.assertIn("sha256sum --check", script)
        self.assertIn("mv -f", script)
        self.assertIn("kilobyte.gguf", MODEL_URL)  # the custom brain release asset

    def test_service_uses_one_daemon(self):
        unit = (Path(__file__).parents[1] / "systemd" / "kilobyte.service").read_text()
        self.assertIn("kilobyte.daemon", unit)
        self.assertIn("Restart=on-failure", unit)

    def test_service_is_enabled_for_boot(self):
        install = (Path(__file__).parents[1] / "scripts" / "install.sh").read_text()
        self.assertIn("systemctl enable kilobyte.service", install)
        unit = (Path(__file__).parents[1] / "systemd" / "kilobyte.service").read_text()
        self.assertIn("WantedBy=multi-user.target", unit)

    def test_installers_agree_on_the_service_account(self):
        """The unit hardcodes a user while the installers choose one. If they disagree,
        the service runs as one account with its data owned by another and every write
        fails -- which only shows up on a machine where the login name differs."""
        if not MODEL_URL:
            self.skipTest("kilobyte-framework has no brain installer to agree with")
        scripts = Path(__file__).parents[1] / "scripts"
        for name in ("install.sh", "install-online.sh", "install-model.sh"):
            text = (scripts / name).read_text()
            self.assertIn('KILOBYTE_USER:-kilobyte}"', text, f"{name} defaults to a different account")
            self.assertNotIn("SUDO_USER", text, f"{name} still derives the account from the invoking user")

    def test_install_rewrites_the_unit_for_the_chosen_account(self):
        install = (Path(__file__).parents[1] / "scripts" / "install.sh").read_text()
        self.assertIn("s/^User=.*/User=$KILO_USER/", install)
        self.assertIn("s/^Group=.*/Group=$KILO_GROUP/", install)

    def test_online_installer_bootstraps_this_repository(self):
        script = (Path(__file__).parents[1] / "scripts" / "install-online.sh").read_text()
        self.assertIn("citadelconsortium/kilobyte-framework", script)
        self.assertNotIn("citadelconsortium/kilobyte}", script)


if __name__ == "__main__":
    unittest.main()
